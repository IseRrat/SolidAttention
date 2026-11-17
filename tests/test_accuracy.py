from array import array
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import accuracy
import accuracy_client
import experiment_common as common
import test_experiments as fixtures


def logits_fixture(root):
    values=array("f",[1000,999,-1000,-1000])
    if sys.byteorder!="little": values.byteswap()
    (root/"logits.f32").write_bytes(values.tobytes())
    result={"logits":{"file":"logits.f32","dtype":"float32-le","rows":2,"columns":2,"sha256":common.digest(root/"logits.f32")},
            "generated_token_ids":[0,0],"decode_input_token_ids":[0]}
    fixtures.write(root/"result.json",result)
    return root/"result.json"


class AccuracyTests(unittest.TestCase):
    def test_teacher_forced_nll_is_stable_and_matches_target_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=logits_fixture(Path(temporary))
            expected=(math.log1p(math.exp(-1))+math.log(2))/2
            self.assertAlmostEqual(accuracy_client.mean_nll(path,[0,1]),expected)
            self.assertAlmostEqual(accuracy_client.mean_nll(path,[0,1],3),expected*2/3)
            with self.assertRaisesRegex(ValueError,"targets differ"): accuracy_client.mean_nll(path,[1,0])
            with self.assertRaises(ValueError): accuracy_client.mean_nll(path,[0,2])
            (path.parent/"logits.f32").write_bytes(b"changed")
            with self.assertRaises(ValueError): accuracy_client.mean_nll(path,[0,1])

    def test_middle_truncation_keeps_prefix_suffix_and_rejects_implicit_cuts(self):
        self.assertEqual(accuracy_client.truncate([1,2,3,4,5,6],4,"middle"),[1,2,5,6])
        self.assertEqual(accuracy_client.truncate([1,2,3],1,"middle"),[1])
        with self.assertRaises(ValueError): accuracy_client.truncate([1,2,3],2,"error")

    def test_dataset_preparation_pins_template_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/"source.jsonl"; template=root/"template.txt"
            template.write_text("Context: {context}\nQuestion: {input}\nAnswer:",encoding="utf-8")
            row={"_id":"sample-a","context":"synthetic","input":"which?","answers":["one"]}
            source.write_text(json.dumps(row)+"\n",encoding="utf-8")
            manifest=accuracy.prepare(source,template,root/"prepared","qasper","pinned-data")
            _,samples=accuracy.read_samples(manifest)
            self.assertEqual(samples[0]["id"],"sample-a")
            self.assertIn("synthetic",samples[0]["prompt"])
            source.write_text((json.dumps(row)+"\n")*2,encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"coverage|checksum"): accuracy.read_samples(manifest)
            with self.assertRaisesRegex(ValueError,"duplicate"): accuracy.prepare(source,template,root/"duplicate","qasper","pinned-data")

    def test_accuracy_export_covers_every_sample_and_audits_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); _,template=fixtures.fixture(root)
            source=root/"source.jsonl"; source.write_text(json.dumps({"id":"one","context":"test","answers":["prediction"]})+"\n")
            prompt=root/"prompt-template.txt"; prompt.write_text("{context}")
            manifest=accuracy.prepare(source,prompt,root/"dataset","qasper","revision")
            run_path,_=fixtures.write_run(root/"template.json")
            record={"result":str(run_path),"result_sha256":common.digest(run_path),"manifest_sha256":common.digest(run_path.parent/"manifest.json")}
            predictions=root/"predictions.json"
            fixtures.write(predictions,{"dataset_sha256":common.digest(manifest),"predictions":{"one":{"pred":"prediction","run":record}}})
            output=accuracy.export(manifest,predictions,root/"export")
            row=json.loads((output/"qasper.jsonl").read_text())
            self.assertEqual(row["answers"],["prediction"])
            state=common.load(predictions); state["predictions"]["one"]["pred"]="different"; fixtures.write(predictions,state)
            with self.assertRaisesRegex(ValueError,"prediction differs"): accuracy.export(manifest,predictions,root/"bad")

    def test_official_scorer_is_revision_locked_and_receives_exported_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); predictions=root/"pred"; predictions.mkdir(); repository=root/"official"; official=repository/"LongBench"; official.mkdir(parents=True)
            (official/"eval.py").write_text("def scorer(dataset, predictions, answers, classes):\n    assert dataset == 'qasper'\n    return 100.0 if predictions[0] in answers[0] else 0.0\n")
            (official/"metrics.py").write_text("# synthetic interface fixture\n")
            file=predictions/"qasper.jsonl"; file.write_text(json.dumps({"pred":"ok","answers":["ok"],"all_classes":[]})+"\n")
            fixtures.write(predictions/"export.json",{"files":{"qasper.jsonl":common.digest(file)},"provenance":["code","runner"],"model_identity":{}})
            revision="a"*40
            def git(argv,**kwargs): return types.SimpleNamespace(stdout=revision if "rev-parse" in argv else "")
            with mock.patch.object(accuracy.subprocess,"run",git):
                result=accuracy.score_longbench(predictions,repository,revision,root/"score.json")
            self.assertEqual(result["scores_percent"],{"qasper":100.0})
            with self.assertRaises(ValueError): accuracy.score_longbench(predictions,repository,"main",root/"bad.json")

    def test_opencompass_generation_and_ppl_contract(self):
        modules={key:types.ModuleType(key) for key in ("opencompass","opencompass.models","opencompass.models.base","opencompass.registry","numpy")}
        class Base:
            def __init__(self,**kwargs): pass
        modules["opencompass.models.base"].BaseModel=Base
        modules["opencompass.registry"].MODELS=types.SimpleNamespace(register_module=lambda:lambda cls:cls)
        modules["numpy"].asarray=lambda values:values
        path=Path(__file__).resolve().parents[1]/"scripts/opencompass_model.py"
        spec=importlib.util.spec_from_file_location("solid_oc_fixture",path); module=importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules,modules): spec.loader.exec_module(module)
        client=mock.Mock(); client.generate.return_value=({"text":"answer"},{})
        client.tokenize.return_value=[1,2]; client.score.return_value=(.25,{})
        with mock.patch.object(module,"RunnerClient",return_value=client):
            model=module.SolidAttentionModel("config","runner","output")
        self.assertEqual(model.generate(["rendered prompt"],8),["answer"])
        self.assertEqual(model.get_ppl(["a","b"],[1,2]),[.25,.25])
        self.assertEqual(client.score.call_args_list,[mock.call("a",1),mock.call("b",2)])
        self.assertEqual(model.get_token_len("prompt"),2)
        with self.assertRaises(ValueError): model.generate(["x"],1,temperature=.8)
        with self.assertRaises(ValueError): model.get_ppl(["x"],[1,2])

    def test_prediction_exports_merge_with_one_model_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); sources=[]
            for dataset in ("qasper","triviaqa"):
                folder=root/dataset; folder.mkdir(); path=folder/(dataset+".jsonl"); path.write_text('{}\n')
                fixtures.write(folder/"export.json",{"provenance":["code","runner"],"model_identity":{"kv_dtype":"f16"},"files":{path.name:common.digest(path)}})
                sources.append(folder)
            output=accuracy.merge_exports(sources,root/"combined")
            self.assertEqual(set(common.load(output/"export.json")["files"]),{"qasper.jsonl","triviaqa.jsonl"})
            metadata=common.load(sources[1]/"export.json"); metadata["model_identity"]["kv_dtype"]="q4_0"; fixtures.write(sources[1]/"export.json",metadata)
            with self.assertRaisesRegex(ValueError,"different models"): accuracy.merge_exports(sources,root/"bad")

    def test_official_metrics_ignore_module_cache_and_restore_import_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"eval.py").write_text("from metrics import score\ndef scorer(*args): return score\n")
            (root/"metrics.py").write_text("score = 37.0\n")
            cached=types.ModuleType("metrics"); cached.score=-1
            old_path=sys.path[:]; old_bytecode=sys.dont_write_bytecode
            with mock.patch.dict(sys.modules,{"metrics":cached}):
                with accuracy.official_longbench(root) as (official,hashes):
                    self.assertEqual(official.scorer(),37)
                    self.assertEqual(hashes["metrics.py"],common.digest(root/"metrics.py"))
                self.assertIs(sys.modules["metrics"],cached)
            self.assertEqual(sys.path,old_path)
            self.assertEqual(sys.dont_write_bytecode,old_bytecode)
            self.assertFalse((root/"__pycache__").exists())


if __name__=="__main__": unittest.main()
