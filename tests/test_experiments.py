import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import experiments
import experiment_common as common
import experiment_runtime as runtime
import plot_experiments


def write(path,value): common.atomic_json(Path(path),value)


def fixture(root):
    prompt = root/"corpus.txt"; prompt.write_text("synthetic corpus",encoding="utf-8")
    config = {"schema_version":1,"model":str(root/"model.gguf"),"model_sha256":"a"*64,"prompt":str(prompt),
              "output":str(root/"unused/result.json"),"context_tokens":32,"batch_tokens":2,"microbatch_tokens":1,
              "max_new_tokens":3,"threads":1,"gpu_layers":0,"flash_attention":True,"kv_on_gpu":True,"ignore_eos":False,
              "sampling":"greedy","kv_dtype":"f16"}
    write(root/"template.json",config)
    write(root/"tokens.json",{"schema_version":1,"source_sha256":common.digest(prompt),"model_parts_sha256":["a"*64],
                              "add_special":True,"parse_special":True,"token_ids":[7,8,9,10,11,12,13,14]})
    spec = {"schema_version":1,"models":[{"name":"model","config":"template.json","tokens":"tokens.json","corpus":"corpus.txt"}],
            "conditions":[{"name":"unit","session_root":str(root/"sessions"),"cache_state":"os-managed", "memory_max_bytes":None,"cpus":[],"background":None}],
            "repeats":2,"warmups":1,"seed":13,"max_new_tokens":3,
            "suites":[{"name":"tiny","lengths":[4],"variants":["origin","solid"],"blocks":[2],"budgets":[0],"conditions":["unit"]}]}
    write(root/"spec.json",spec)
    return root/"spec.json",config


def write_run(source, runner_sha="runner", code="commit", timings=(2,4)):
    config = common.absolute_config(source)
    output = Path(config["output"]); folder = output.parent; folder.mkdir(parents=True)
    (folder/"prompt.txt").write_bytes(Path(config["prompt"]).read_bytes())
    effective = copy.deepcopy(config); effective["prompt"] = str(folder/"prompt.txt")
    write(folder/"config.json",effective)
    tokens = [0]*(len(timings)+1)
    trace = [{"index":0,"token_id":0,"source":"prefill","total_ms":1,"process_ms":0,"sample_ms":1}]
    trace += [{"index":i+1,"token_id":0,"source":"decode","total_ms":value,"process_ms":value-1,"sample_ms":1} for i,value in enumerate(timings)]
    value = {"model_parts_sha256":common.model_hashes(config),"prompt_token_ids":config.get("prompt_token_ids",[7,8]),
             "generated_token_ids":tokens,"decode_input_token_ids":[0]*len(timings),"text":"prediction","kv_dtype":config["kv_dtype"],
             "trace":trace,"stop_reason":"length","upstream_commit":"upstream","build_info":{"sources":"synthetic"},
             "timing":{"decode_steps":len(timings),"decode_loop_ms":sum(timings),"prefill_ms":10,"first_token_ms":11,"request_ms":20}}
    if "attention" in config:
        value["attention"] = dict(config["attention"],context_budget_tokens=config["attention"].get("context_budget_tokens",1))
    write(output,value)
    manifest = {"status":"complete","returncode":0,"code_commit":code,"runner_sha256":runner_sha,"upstream_commit":"upstream",
                "build_info":value["build_info"],"result_sha256":common.digest(output),"source_config_sha256":common.digest(Path(source)),
                "effective_config_sha256":common.digest(folder/"config.json"),"prompt_sha256":common.digest(folder/"prompt.txt"),"sampled_peak_rss_bytes":4096}
    write(folder/"manifest.json",manifest)
    return output,value


class ExperimentTests(unittest.TestCase):
    def test_fixed_decode_inputs_require_matching_mode_tokens_and_length(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);_,config=fixture(root)
            config.update(output=str(root/"run/result.json"),max_new_tokens=3,replay_token_ids=[0,0])
            source=root/"fixed.json";write(source,config)
            output,value=write_run(source)
            value.update(stop_reason="replay_length",replay=True)
            row={"result":str(output),"config":str(source),"config_sha256":common.digest(source)}
            def save():
                write(output,value);manifest=common.load(output.parent/"manifest.json")
                manifest["result_sha256"]=common.digest(output);write(output.parent/"manifest.json",manifest)
            save();experiments.audited_job(row,{},root)
            value["decode_input_token_ids"]=[0,1];save()
            with self.assertRaisesRegex(ValueError,"input tokens differ"):experiments.audited_job(row,{},root)
            value["decode_input_token_ids"]=[0,0];value["stop_reason"]="length";save()
            with self.assertRaisesRegex(ValueError,"fixed output length"):experiments.audited_job(row,{},root)

    def test_coalescing_ablation_changes_only_adjacent_upload_merging(self):
        template={"attention":{"infllm_attention":True,"selection_backend":"cuda","direct_decode_reads":True}}
        on=experiments.variant_config(template,"solid",32,1024)
        off=experiments.variant_config(template,"solid-no-coalescing",32,1024)
        self.assertTrue(on["attention"].pop("coalesce_uploads"))
        self.assertFalse(off["attention"].pop("coalesce_uploads"))
        self.assertEqual(on,off)
        self.assertEqual(off["attention"]["layout"],"interleaved")
        self.assertEqual(off["attention"]["schedule"],"overlap+sync_reuse")

    def test_scheduler_ablation_changes_only_schedule(self):
        template={"attention":{"infllm_attention":True,"selection_backend":"cuda","direct_decode_reads":True}}
        configs=[]
        for variant,mode in (("prefetch","naive"),("overlap","overlap"),("solid","overlap+sync_reuse")):
            config=experiments.variant_config(template,variant,32,1024)
            self.assertEqual(config["attention"].pop("schedule"),mode)
            self.assertTrue(config["attention"]["prefetch_ahead"])
            self.assertTrue(config["attention"]["fuse_kv_projection"])
            self.assertEqual(config["attention"]["layout"],"interleaved")
            configs.append(config)
        self.assertEqual(configs[0],configs[1])
        self.assertEqual(configs[1],configs[2])

    def test_interleaving_ablation_keeps_prefetch_scoring_and_projection(self):
        template={"attention":{"infllm_attention":True,"selection_backend":"cuda","direct_decode_reads":True}}
        full=experiments.variant_config(template,"solid",32,1024)
        separate=experiments.variant_config(template,"solid-no-interleaved",32,1024)
        self.assertEqual(separate["attention"].pop("layout"),"separate")
        self.assertEqual(full["attention"].pop("layout"),"interleaved")
        self.assertEqual(separate,full)
        self.assertTrue(separate["attention"]["fuse_kv_projection"])
        self.assertTrue(separate["attention"]["prefetch_ahead"])

    def test_prefetch_ablation_changes_only_advance_reads(self):
        template = {"attention": {"infllm_attention":True, "selection_backend":"cuda", "direct_decode_reads":True}}
        full = experiments.variant_config(template, "solid", 32, 1024)
        demand = experiments.variant_config(template, "solid-no-prefetch", 32, 1024)
        self.assertTrue(full["attention"]["prefetch_ahead"])
        self.assertFalse(demand["attention"].pop("prefetch_ahead"))
        del full["attention"]["prefetch_ahead"]
        self.assertEqual(demand, full)
        self.assertTrue(demand["attention"]["prefetch"])
        self.assertEqual(demand["attention"]["schedule"], "overlap+sync_reuse")

    def test_prefetch_result_matches_requested_advance_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source, template=fixture(root)
            config=experiments.variant_config(template,"solid-no-prefetch",32,0)
            config["output"]=str(root/"run/result.json")
            write(root/"config.json",config)
            output,result=write_run(root/"config.json")
            result["prefetch"]={"ahead_enabled":False}
            write(output,result)
            manifest=common.load(output.parent/"manifest.json")
            manifest["result_sha256"]=common.digest(output)
            write(output.parent/"manifest.json",manifest)
            common.audit_run(output)
            result["prefetch"]["ahead_enabled"]=True
            write(output,result)
            manifest["result_sha256"]=common.digest(output)
            write(output.parent/"manifest.json",manifest)
            with self.assertRaisesRegex(ValueError,"prefetch-ahead"):
                common.audit_run(output)

    def test_paired_sparse_variants_share_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            spec_path,_=fixture(root)
            spec=common.load(spec_path)
            suite=spec["suites"][0]
            suite.update(lengths=[4,8],variants=["offload","offload-sparse","solid"],
                         offload_sparse_budgets={"4":"full","8":4})
            write(spec_path,spec)
            with self.assertRaisesRegex(ValueError, "same budget"):
                experiments.create(spec_path,root/"invalid-plan")
            del suite["offload_sparse_budgets"]
            write(spec_path,spec)
            _,plan=experiments.read_plan(experiments.create(spec_path,root/"plan"))
            for row in plan["jobs"]:
                attention=common.load(row["config"])["attention"]
                self.assertNotIn("context_budget_tokens",attention)

    def test_worker_cpu_settings_are_applied_only_to_prefetch_variants(self):
        template={"attention":{"io_cpu":6,"coordinator_cpu":7}}
        for variant in experiments.VARIANTS:
            config=experiments.variant_config(template,variant,32,1024)
            a=config.get("attention",{})
            if a.get("prefetch"):
                self.assertEqual((a["io_cpu"],a["coordinator_cpu"]),(6,7))
            else:
                self.assertNotIn("io_cpu",a); self.assertNotIn("coordinator_cpu",a)

    def test_prefill_policy_survives_dense_and_sparse_variants(self):
        template={"attention":{"prefill_backend":"cpu","prefill_tile_tokens":128}}
        for variant in experiments.VARIANTS:
            config=experiments.variant_config(template,variant,32,1024)
            if "attention" in config:
                self.assertEqual(config["attention"]["prefill_backend"],"cpu")
                self.assertEqual(config["attention"]["prefill_tile_tokens"],128)
        for bad in ({"prefill_backend":"unknown"},{"prefill_tile_tokens":0},{"prefill_tile_tokens":True}):
            with self.assertRaises(ValueError): experiments.variant_config({"attention":bad},"offload",32,1024)
        infllm = {"attention": {"infllm_attention":True,"prefill_backend":"cuda"}}
        for variant in ("offload-sparse", "solid"):
            attention = experiments.variant_config(infllm, variant, 32, 0)["attention"]
            self.assertEqual(attention["prefill_backend"], "cuda")
            self.assertNotIn("context_budget_tokens", attention)

    def test_selection_backend_survives_sparse_variant_preparation(self):
        for selector in ("auto", "cpu", "cuda"):
            template = {"attention": {"selection_backend": selector}}
            for variant in experiments.VARIANTS:
                config = experiments.variant_config(template,variant,32,1024)
                if config.get("attention",{}).get("sparse"):
                    self.assertEqual(config["attention"]["selection_backend"],selector)
                else:
                    self.assertNotIn("selection_backend",config.get("attention",{}))
        with self.assertRaisesRegex(ValueError,"selection backend"):
            experiments.variant_config({"attention":{"selection_backend":"unknown"}},"solid",32,1024)

    def test_exact_prefix_matrix_repeats_and_disabled_traces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); spec,_=fixture(root)
            path=experiments.create(spec,root/"plan")
            _,plan=experiments.read_plan(path)
            self.assertEqual(len(plan["jobs"]),6)
            self.assertTrue(all(row["warmup"] for row in plan["jobs"][:2]))
            for row in plan["jobs"]:
                config=common.load(row["config"])
                self.assertEqual(config["prompt_token_ids"],[7,8,9,10])
                self.assertEqual(config["context_tokens"],6)
                self.assertTrue(config["ignore_eos"])
                if "attention" in config:
                    self.assertFalse(config["attention"]["selection_trace"])
                    self.assertFalse(config["attention"]["schedule_trace"])
            self.assertFalse((root/"plan/runs").exists())

    def test_invalid_corpus_and_duplicate_axes_leave_no_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); spec,_=fixture(root)
            original=common.load(spec)
            for bad in (dict(original,repeats=0),dict(original,max_new_tokens=True)):
                write(spec,bad)
                with self.assertRaises(ValueError): experiments.create(spec,root/"bad")
            write(spec,original)
            bundle=common.load(root/"tokens.json"); bundle["model_parts_sha256"]=["b"*64]; write(root/"tokens.json",bundle)
            with self.assertRaisesRegex(ValueError,"identity"): experiments.create(spec,root/"bad")
            self.assertFalse((root/"bad").exists())

    def test_collection_preserves_missing_and_failed_repetitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); spec,_=fixture(root); path=experiments.create(spec,root/"plan")
            plan=common.load(path); measured=[row for row in plan["jobs"] if not row["warmup"]]
            row=measured[0]; write_run(row["config"])
            write(path.parent/"execution.json",{"plan_sha256":common.digest(path),"runner_sha256":"runner",
                  "jobs":{row["id"]:{"status":"complete"},measured[1]["id"]:{"status":"failed"}}})
            summary=experiments.collect(path,root/"report")
            self.assertEqual(sum(group["complete"] for group in summary["groups"]),1)
            self.assertEqual(sum(group["expected"] for group in summary["groups"]),4)
            run=next(run for run in summary["runs"] if run["status"]=="complete")
            self.assertAlmostEqual(run["metrics"]["decode_tokens_per_second"],1000/3)
            self.assertEqual(run["metrics"]["decode_latency_ms"]["median"],3)
            self.assertEqual(run["metrics"]["decode_latency_ms"]["p99"],3.98)
            with self.assertRaisesRegex(ValueError,"all planned"): plot_experiments.render(root/"report/summary.json",root/"figure","tiny")
            result=common.load(row["result"]); result["prompt_token_ids"][0]=999; write(row["result"],result)
            with self.assertRaisesRegex(ValueError,"checksum"): experiments.collect(path,root/"bad-report")

    def test_changed_plan_config_and_mixed_binaries_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); spec,_=fixture(root); path=experiments.create(spec,root/"plan"); plan=common.load(path)
            rows=[row for row in plan["jobs"] if not row["warmup"]][:2]
            write_run(rows[0]["config"]); write_run(rows[1]["config"],runner_sha="other")
            write(path.parent/"execution.json",{"plan_sha256":common.digest(path),"runner_sha256":"runner",
                  "jobs":{row["id"]:{"status":"complete"} for row in rows}})
            with self.assertRaisesRegex(ValueError,"identities|runner"): experiments.collect(path,root/"bad-report")
            config=common.load(rows[0]["config"]); config["max_new_tokens"]=5; write(rows[0]["config"],config)
            with self.assertRaisesRegex(ValueError,"checksum"): experiments.read_plan(path)

    @unittest.skipUnless(sys.platform.startswith("linux"),"external executor uses POSIX file locks")
    def test_executor_resume_runs_only_pending_jobs_with_fake_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); spec,_=fixture(root); path=experiments.create(spec,root/"plan")
            runner=root/"fake-runner"; runner.write_bytes(b"synthetic executable identity")
            class FakeProcess:
                returncode=0
                def __init__(self,argv,**kwargs): write_run(argv[2],common.digest(runner))
                def poll(self): return self.returncode
            with mock.patch.object(experiments.subprocess,"Popen",FakeProcess):
                first=experiments.execute(path,runner,limit=1)
                self.assertEqual(len(first["jobs"]),1)
                second=experiments.execute(path,runner)
                self.assertEqual(len(second["jobs"]),6)
                final=experiments.execute(path,runner)
                self.assertEqual(final,second)
            self.assertTrue(all(row["status"]=="complete" for row in final["jobs"].values()))
            self.assertTrue(all(group["complete"]==2 for group in experiments.collect(path,root/"report")["groups"]))

    def test_fio_commands_are_read_only_and_protect_existing_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); data=root/"load file"; data.write_bytes(b"x"*8192)
            settings={"name":"io","session_root":str(root),"cache_state":"os-managed","background":
                      {"kind":"random-read","file":str(data),"rate":200000,"iodepth":64}}
            command=runtime.fio_command(settings,root)
            self.assertIn("--readonly",command); self.assertIn("--allow_file_create=0",command)
            self.assertIn("--rate_iops=200000",command)
            self.assertIn("--filename="+str(data.resolve()),command)
            with self.assertRaisesRegex(ValueError,"overlaps"): runtime.fio_command(settings,root,[data])
            settings["background"]["kind"]="write"
            with self.assertRaises(ValueError): runtime.fio_command(settings,root)

    def test_resource_limits_and_statistics_keep_their_units(self):
        settings={"name":"unit","session_root":"/tmp/kv","cache_state":"os-managed","memory_max_bytes":17179869184,"swap_max_bytes":0,"cpus":[1,3]}
        command=runtime.isolated_command(["python3","runner.py"],settings)
        self.assertIn("--property=MemoryMax=17179869184",command)
        self.assertEqual(command[-5:],["taskset","--cpu-list","1,3","python3","runner.py"])
        self.assertEqual(common.distribution([]),None)
        self.assertEqual(common.distribution([7])["p999"],7)
        with self.assertRaises(ValueError): common.distribution([float("nan")])

    def test_fio_status_stream_keeps_complete_samples_and_final_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"fio.json"
            first={"jobs":[{"read":{"total_ios":3}}]}; final={"jobs":[{"error":0,"read":{"total_ios":30}}]}
            path.write_text("Starting thread\n"+json.dumps(first)+"\n"+json.dumps(final)+"\n{\"jobs\":")
            self.assertEqual(runtime.fio_observations(path),[first,final])

    @unittest.skipUnless(sys.platform.startswith("linux"),"external executor uses POSIX file locks")
    def test_executor_rejects_short_output_before_marking_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); spec,_=fixture(root); path=experiments.create(spec,root/"plan")
            runner=root/"fake-runner"; runner.write_bytes(b"synthetic runner")
            class FakeProcess:
                returncode=0
                def __init__(self,argv,**kwargs): write_run(argv[2],common.digest(runner),timings=(2,))
                def poll(self): return 0
            with mock.patch.object(experiments.subprocess,"Popen",FakeProcess):
                state=experiments.execute(path,runner,limit=1)
            record=next(iter(state["jobs"].values()))
            self.assertEqual(record["status"],"failed")
            self.assertIn("fixed output length",record["error"])

    @unittest.skipUnless(sys.platform.startswith("linux"),"external executor uses POSIX file locks")
    def test_resume_rechecks_measurement_validity_before_skipping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); spec,_=fixture(root); path=experiments.create(spec,root/"plan")
            runner=root/"fake-runner"; runner.write_bytes(b"synthetic runner")
            row=common.load(path)["jobs"][0]
            write_run(row["config"],common.digest(runner),timings=(2,))
            write(path.parent/"execution.json",{"plan_sha256":common.digest(path),"runner_sha256":common.digest(runner),"jobs":{row["id"]:{"status":"complete"}}})
            with mock.patch.object(experiments.subprocess,"Popen") as spawn:
                with self.assertRaisesRegex(ValueError,"fixed output length"): experiments.execute(path,runner)
            spawn.assert_not_called()

    def test_audit_requires_applied_limits_and_nonempty_interference_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); spec,_=fixture(root); path=experiments.create(spec,root/"plan")
            row=common.load(path)["jobs"][0]; result,_=write_run(row["config"])
            settings={"name":"limited","session_root":str(root),"cache_state":"os-managed","memory_max_bytes":1024,"swap_max_bytes":None}
            self.assertIn("--property=MemorySwapMax=0",runtime.isolated_command(["synthetic"],settings))
            with self.assertRaisesRegex(ValueError,"observed cgroup"): experiments.audited_job(row,settings,root)
            manifest=common.load(result.parent/"manifest.json"); manifest["cgroup_end"]={"memory.max":"1024","memory.swap.max":"0"}; write(result.parent/"manifest.json",manifest)
            settings["background"]={"kind":"random-read","file":str(root/"load.data"),"rate":1,"iodepth":1}
            for jobs in ([],[{"error":0,"read":{"total_ios":0}}],[{"error":5,"read":{"total_ios":1}}]):
                write(root/"fio.json",{"jobs":jobs})
                with self.assertRaisesRegex(ValueError,"fio report"): experiments.audited_job(row,settings,root)
            write(root/"fio.json",{"jobs":[{"error":0,"read":{"total_ios":1}}]})
            self.assertEqual(experiments.audited_job(row,settings,root)[-1],[{"total_ios":1}])
            with self.assertRaisesRegex(ValueError,"requires memory_max_bytes"):
                runtime.condition(dict(settings,memory_max_bytes=None,swap_max_bytes=0))


if __name__=="__main__": unittest.main()
