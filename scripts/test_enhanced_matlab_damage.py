"""Controlled damage-evidence run on declared train/validation samples only."""
from __future__ import annotations

import argparse, base64, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
MATLAB_ROOT = ROOT / "matlab"
MANIFEST = ROOT / "evaluation" / "outputs" / "dataset_manifest.csv"
SAMPLES = ROOT / "config" / "development_damage_samples.json"
OUTPUT = ROOT / "outputs" / "matlab_damage_test"
SEGMENTATION_BEFORE = ROOT / "outputs" / "matlab_segmentation_test"
IMAGE_MAP = {
    "original":"original.png", "papaya_mask_clean":"clean_mask.png", "papaya_mask_inner":"inner_mask.png",
    "stem_mask":"stem_mask.png", "shadow_mask":"shadow_mask.png", "reflection_mask":"reflection_mask.png",
    "brown_decay_mask":"brown_decay_mask.png", "dark_decay_mask":"dark_decay_mask.png",
    "white_mold_mask":"white_mold_mask.png", "lesion_mask":"lesion_mask.png",
    "abnormal_texture_mask":"abnormal_texture_mask.png", "filtered_damage_mask":"filtered_damage_mask.png",
    "damage_highlight":"damage_highlight.png",
}
COMPARISON_COLUMNS = [
    "sample","expected_class","ai_class","matlab_class_before","matlab_class_after",
    "damage_before","damage_after","healthy_before","healthy_after","stem_excluded",
    "shadow_excluded","reflection_excluded","agreement_before","agreement_after",
    "review_required_before","review_required_after",
]


def manifest_rows():
    with MANIFEST.open("r",encoding="utf-8-sig",newline="") as handle:
        return {str(Path(r["source_path"]).resolve()).lower():r for r in csv.DictReader(handle)}


def load_samples():
    lookup=manifest_rows(); declared=json.loads(SAMPLES.read_text(encoding="utf-8"))["samples"]; result=[]
    for entry in declared:
        source=(ROOT/entry["source_path"]).resolve(); row=lookup.get(str(source).lower())
        if row is None: raise ValueError(f"Sample absent from current manifest: {source}")
        if row["split"] not in {"train","validation"}: raise ValueError(f"Held-out test sample forbidden: {source}")
        if row["split"] != entry["manifest_split"] or row["class_name"] != entry["expected_class"]:
            raise ValueError(f"Manifest metadata mismatch: {source}")
        result.append({**entry,"source":source,"sha256":row["sha256"]})
    counts={name:sum(s["expected_class"]==name for s in result) for name in ("Fresh","Unripe","Rotten")}
    if counts != {"Fresh":8,"Unripe":8,"Rotten":8}: raise ValueError(f"Expected exactly 8 per class, got {counts}")
    return result


def previous_results():
    indexed={}
    for path in SEGMENTATION_BEFORE.glob("*/result.json"):
        data=json.loads(path.read_text(encoding="utf-8")); source=data.get("source_image_path")
        if source: indexed[str(Path(source).resolve()).lower()]=data
    return indexed


def save_sample(result, ai, sample, folder):
    folder.mkdir(parents=True,exist_ok=True); images=result.get("images") or {}
    for key,filename in IMAGE_MAP.items():
        if images.get(key): (folder/filename).write_bytes(base64.b64decode(images[key],validate=True))
    report={key:value for key,value in result.items() if key != "images"}
    report["development_sample"]={k:str(v) if isinstance(v,Path) else v for k,v in sample.items()}
    report["development_ai_comparison"]=ai
    (folder/"result.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


def review_required(ai, matlab, agreement):
    reliability=matlab.get("measurement_reliability") or {}
    return agreement is False or reliability.get("matlab_class_reliable") is not True or bool(ai.get("uncertainty_warning"))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--preflight-only",action="store_true"); args=parser.parse_args()
    try:
        import matlab.engine
        from web.services.prediction_service import PredictionService
        samples=load_samples()
        configs=[json.loads((ROOT/"config"/name).read_text(encoding="utf-8")) for name in ("model_config.json","model_registry.json","web_config.json")]
        predictor=PredictionService(*configs)
        if args.preflight_only:
            print("Damage calibration samples: 24 (8 Fresh, 8 Unripe, 8 Rotten)")
            print("Manifest validation: PASS (train/validation only)"); print("MATLAB and FreshSight imports: PASS"); return 0
    except Exception as exc:
        print(f"ERROR: {exc}",file=sys.stderr); return 1
    engine=None; comparisons=[]; before=previous_results()
    try:
        engine=matlab.engine.start_matlab(); engine.addpath(engine.genpath(str(MATLAB_ROOT)),nargout=0)
        print("=== FreshSight Controlled MATLAB Damage Test ===")
        print("Development data only; no threshold changes or held-out evaluation.")
        for index,sample in enumerate(samples,1):
            result=json.loads(str(engine.run_freshsight_api(str(sample["source"]),nargout=1)))
            ai=predictor.predict(str(sample["source"])); rel=result.get("measurement_reliability") or {}
            agreement=ai.get("predicted_class")==result.get("rule_class") if ai.get("available") and rel.get("matlab_class_reliable") else None
            ai_record={"predicted_class":ai.get("predicted_class"),"confidence":ai.get("confidence"),"device":ai.get("device"),"agreement":agreement,"uncertainty_warning":ai.get("uncertainty_warning")}
            folder=OUTPUT/f"{index:02d}_{sample['expected_class']}_{sample['source'].stem}"; save_sample(result,ai_record,sample,folder)
            evidence=result.get("damage_evidence") or {}; old=before.get(str(sample["source"]).lower())
            if old:
                old_ai=old.get("development_ai_comparison") or {}; old_agreement=old_ai.get("agreement")
                comparisons.append({
                    "sample":sample["source"].name,"expected_class":sample["expected_class"],"ai_class":ai.get("predicted_class"),
                    "matlab_class_before":old.get("rule_class"),"matlab_class_after":result.get("rule_class"),
                    "damage_before":old.get("damage_percentage"),"damage_after":result.get("damage_percentage"),
                    "healthy_before":old.get("healthy_percentage"),"healthy_after":result.get("healthy_percentage"),
                    "stem_excluded":evidence.get("excluded_stem_percentage"),"shadow_excluded":evidence.get("excluded_shadow_percentage"),
                    "reflection_excluded":evidence.get("excluded_reflection_percentage"),"agreement_before":old_agreement,"agreement_after":agreement,
                    "review_required_before":old_agreement is not True,"review_required_after":review_required(ai,result,agreement),
                })
            print(f"\n[{index}/24] {sample['source'].name}")
            print(f"Expected/AI/MATLAB: {sample['expected_class']} / {ai.get('predicted_class')} ({ai.get('confidence')}) / {result.get('rule_class')}")
            print(f"Expected damage level: {sample['expected_damage_level']}")
            print(f"Measured damage/severity: {result.get('damage_percentage')} / {result.get('damage_severity')}")
            for key in ("excluded_stem_percentage","excluded_shadow_percentage","excluded_reflection_percentage","brown_decay_percentage","dark_decay_percentage","mold_percentage","lesion_percentage","abnormal_texture_percentage"):
                print(f"{key}: {evidence.get(key)}")
            print(f"Agreement: {agreement}; manual review: {review_required(ai,result,agreement)}")
        OUTPUT.mkdir(parents=True,exist_ok=True)
        with (OUTPUT/"damage_refinement_comparison.csv").open("w",encoding="utf-8-sig",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=COMPARISON_COLUMNS); writer.writeheader(); writer.writerows(comparisons)
        print(f"\nOutputs: {OUTPUT}"); print(f"Before/after rows: {len(comparisons)}"); print("Result: COMPLETED — visually inspect evidence masks before claiming improvement.")
        return 0
    except Exception as exc:
        print(f"ERROR: controlled damage test failed: {exc}",file=sys.stderr); return 1
    finally:
        if engine is not None:
            try: engine.quit(); print("MATLAB Engine closed cleanly.")
            except Exception as exc: print(f"WARNING: MATLAB close failed: {exc}",file=sys.stderr)


if __name__ == "__main__": raise SystemExit(main())
