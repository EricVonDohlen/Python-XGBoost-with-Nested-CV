final_model, selected_features, oof_preds, fold_aucs = train_ultimate_xgb(
    csv_path="data.csv",
    target_col="onus_target",
    id_col="accountid",
    pipeline_yaml_path="pipeline_settings.yaml",
    param_yaml_path="xgb_param_ranges.yaml",
    save_dir="ultimate_xgb_pipeline"
)