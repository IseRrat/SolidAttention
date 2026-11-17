from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import accuracy_report
import experiment_common as common
import test_experiments as fixtures


class AccuracyReportTests(unittest.TestCase):
    def test_table_import_uses_official_rows_and_backend_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); _,config=fixtures.fixture(root)
            config["output"]=str(root/"backend/calls/one/run/result.json"); fixtures.write(root/"template.json",config)
            fixtures.write_run(root/"template.json")
            summary=root/"summary.csv"; summary.write_text("dataset,metric,solid-origin\nwino,accuracy,50\narc,accuracy,60\nmmlu,accuracy,70\ngsm,accuracy,80\n")
            mapping=root/"mapping.json"; fixtures.write(mapping,{"wino":"winogrande","arc":"arc-challenge","mmlu":"mmlu","gsm":"gsm8k"})
            longbench=root/"longbench.json"; fixtures.write(longbench,{"scores_percent":{name:40 for name in accuracy_report.LONG_EIGHT},"provenance":["commit","runner"],
                "model_identity":{"model_parts_sha256":["a"*64],"kv_dtype":"f16","sparse":False,"schedule":"naive"}})
            revision="a"*40
            def git(argv,**kwargs): return types.SimpleNamespace(stdout=revision if "rev-parse" in argv else "")
            with mock.patch.object(accuracy_report.subprocess,"run",git):
                output=accuracy_report.import_summary(summary,"solid-origin",mapping,root,revision,root/"template.json",root/"backend",root/"table.json",longbench)
            self.assertEqual(output["average_percent"],60)
            self.assertEqual(output["scores_percent"]["longbench"],40)
            self.assertEqual(len(output["provenance"]["runs"]),1)

    def test_missing_or_duplicate_task_mappings_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); fixtures.fixture(root)
            summary=root/"summary.csv"; summary.write_text("dataset,model\na,50\n")
            mapping=root/"mapping.json"; fixtures.write(mapping,{"a":"mmlu","b":"mmlu"})
            revision="a"*40
            with mock.patch.object(accuracy_report.subprocess,"run",lambda argv,**kwargs:types.SimpleNamespace(stdout=revision if "rev-parse" in argv else "")):
                with self.assertRaisesRegex(ValueError,"one official summary row"):
                    accuracy_report.import_summary(summary,"model",mapping,root,revision,root/"template.json",root,root/"bad.json")


if __name__=="__main__": unittest.main()
