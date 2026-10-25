"""Draft-only native vLLM multi-LoRA registration; never wraps the verifier."""

import torch


class NativeDraftLoRA:
    def __init__(self, draft, metadata, config):
        from vllm.config.lora import LoRAConfig  # noqa: PLC0415
        from vllm.lora.model_manager import create_lora_manager  # noqa: PLC0415

        attention = draft.model.layers[0].self_attn
        self.device = attention.o_proj.weight.device
        self.capacity = config.scheduler_config.max_num_seqs
        self.max_tokens = config.scheduler_config.max_num_batched_tokens
        self.metadata = metadata
        self.resident = {}
        # Separate manager/config: target lora_config and target request IDs stay
        # untouched. Restrict wrappers so shared embeddings/head are never adapted.
        lora = LoRAConfig(
            max_lora_rank=metadata["rank"],
            max_loras=self.capacity,
            max_cpu_loras=self.capacity,
            lora_dtype=torch.bfloat16,
            target_modules=["v_proj", "o_proj"],
        )
        self.manager = create_lora_manager(
            draft,
            max_num_seqs=self.capacity,
            max_num_batched_tokens=self.max_tokens,
            vocab_size=config.speculative_config.draft_model_config.get_vocab_size(),
            lora_config=lora,
            vllm_config=config,
            device=self.device,
        )
        expected = {
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.self_attn.o_proj",
        }
        if set(self.manager.modules) != expected:
            raise ValueError(
                f"Unexpected draft LoRA modules: {set(self.manager.modules)}"
            )
        self.map_tokens([], [])

    def activate(self, requests):
        """Only scheduled requests occupy GPU slots; paused factors stay on CPU."""
        from vllm.lora.lora_model import LoRAModel  # noqa: PLC0415
        from vllm.lora.lora_weights import LoRALayerWeights  # noqa: PLC0415

        if len(requests) > self.capacity:
            raise ValueError("Scheduled draft adapters exceed GPU slot capacity")
        selected = {adapter_id for adapter_id, _ in requests.values()}
        for request_id, adapter_id in list(self.resident.items()):
            if adapter_id not in selected:
                self.manager.remove_adapter(adapter_id)
                del self.resident[request_id]
        for request_id, (adapter_id, factors) in requests.items():
            if request_id in self.resident:
                continue
            layers = {}
            for name, (a, b) in factors.items():
                path = f"model.layers.0.self_attn.{name}"
                # Native packing/scaling mutates B. Preserve the retained CPU
                # originals so resumption cannot scale an adapter a second time.
                layers[path] = LoRALayerWeights(
                    path,
                    self.metadata["rank"],
                    self.metadata["alpha"],
                    a.to(dtype=torch.bfloat16, copy=True),
                    b.to(dtype=torch.bfloat16, copy=True),
                )
            self.manager.add_adapter(
                LoRAModel(adapter_id, self.metadata["rank"], layers)
            )
            self.manager.activate_adapter(adapter_id)
            self.resident[request_id] = adapter_id

    def map_tokens(self, token_ids, request_ids):
        from vllm.lora.layers import LoRAMapping  # noqa: PLC0415

        if len(token_ids) > self.max_tokens:
            raise ValueError("Draft token mapping exceeds its allocated capacity")
        self.manager.set_adapter_mapping(
            LoRAMapping(tuple(token_ids), tuple(request_ids), is_prefill=True)
        )

    def release(self, request_id):
        adapter_id = self.resident.pop(request_id, None)
        if adapter_id is not None:
            self.manager.remove_adapter(adapter_id)
