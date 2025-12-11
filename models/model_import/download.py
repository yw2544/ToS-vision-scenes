from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yw12356/ToS_model_lib",
    repo_type="dataset",
    local_dir="model_lib",
    local_dir_use_symlinks=False
)
