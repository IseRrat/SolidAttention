import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import selection_analysis
import experiment_common as common
import test_experiments as fixtures


class SelectionAnalysisTests(unittest.TestCase):
    def test_selected_overlap_excludes_initial_and_local_regions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); fixtures.fixture(root)
            config=common.load(root/"template.json")
            config["attention"]={"storage":"ssd","sparse":True,"block_tokens":2,"context_budget_tokens":5,"init_tokens":1,"local_tokens":2}
            fixtures.write(root/"template.json",config)
            result_path,value=fixtures.write_run(root/"template.json")
            rows=[{"layer":0,"kv_head":0,"position":9,"ranges":[[0,1],[2,2],[8,2]]},
                  {"layer":0,"kv_head":0,"position":10,"ranges":[[0,1],[2,2],[9,2]]}]
            trace=result_path.parent/"selection.jsonl"; trace.write_text("".join(json.dumps(row)+"\n" for row in rows))
            value["selection_trace_sha256"]=common.digest(trace); fixtures.write(result_path,value)
            manifest=common.load(result_path.parent/"manifest.json"); manifest["result_sha256"]=common.digest(result_path); fixtures.write(result_path.parent/"manifest.json",manifest)
            output=selection_analysis.analyze(result_path)
            self.assertEqual(output["overlap"]["mean"],1)
            self.assertEqual(output["records"][1]["selected_blocks"],1)
            self.assertEqual(output["records"][1]["selected_tokens"],2)
            self.assertEqual(output["records"][1]["local_tokens"],2)

    def test_percentiles_keep_repeat_and_token_counts_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); fixtures.fixture(root)
            path,value=fixtures.write_run(root/"template.json",timings=(1,9))
            metrics=common.run_metrics(value)
            self.assertEqual(metrics["decode_steps"],2)
            self.assertEqual(metrics["output_tokens"],3)
            self.assertEqual(metrics["decode_tokens_per_second"],200)
            self.assertEqual(metrics["decode_latency_ms"]["mean"],5)
            self.assertEqual(metrics["decode_latency_ms"]["p99"],8.92)
            value["timing"]["decode_steps"]=3
            with self.assertRaises(ValueError): common.run_metrics(value)


if __name__=="__main__": unittest.main()
