"""Keep model installation and cache detection wired to the same artifacts."""

from sova import config, install, llama_client


def test_chat_service_uses_configured_context_artifact():
    chat = next(service for service in install.SERVICES if service["name"] == "chat")

    assert chat["hf_repo"] == config.CONTEXT_MODEL_HF_REPO
    assert chat["hf_file"] == config.CONTEXT_MODEL_HF_FILE
    assert llama_client._MODEL_SPECS["com.sova.chat"] == (
        config.CONTEXT_MODEL_HF_REPO,
        config.CONTEXT_MODEL_HF_FILE,
    )

    args = chat["extra_args"]
    assert "--no-mmproj" in args
    assert args[args.index("--parallel") + 1] == "1"
    assert args[args.index("--reasoning-effort") + 1] == "low"
    assert args[args.index("--reasoning-budget") + 1] == "64"
    assert "--spec-type" not in args


def test_installation_has_only_embedding_and_context_models():
    assert [service["name"] for service in install.SERVICES] == ["embedding", "chat"]
    assert set(llama_client._MODEL_SPECS) == {
        "com.sova.embedding",
        "com.sova.chat",
    }
