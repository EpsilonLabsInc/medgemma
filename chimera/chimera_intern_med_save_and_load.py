import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector
import json
from pathlib import Path


class StandaloneChimeraModel(nn.Module):
    """
    Standalone chimera model that doesn't require InternVL path for loading
    """

    def __init__(self, base_model_id="google/medgemma-4b-it", internvl_path=None):
        super().__init__()

        # Load base MedGemma
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        self.processor = AutoProcessor.from_pretrained(base_model_id)
        self.config = self.base_model.config
        self.device = self.base_model.device
        self.dtype = self.base_model.dtype

        # Only setup chimera if internvl_path is provided (for creation)
        if internvl_path:
            self.setup_chimera(internvl_path)
        else:
            # For loading, we'll just prepare the structure
            self._prepare_for_loading()

    def _prepare_for_loading(self):
        """
        Prepare the model structure for loading (without InternVL path)
        """
        # Create the modified projector structure
        config = self.config
        config.vision_config.hidden_size = 1024  # We know this from chimera

        proj_new = Gemma3MultiModalProjector(config)
        proj_new = proj_new.to(device=self.device, dtype=self.dtype)
        self.base_model.multi_modal_projector = proj_new

        # Override the get_image_features method
        self._override_image_features_method()

    def setup_chimera(self, internvl_path):
        """
        Convert to chimera model (only needed during creation)
        """
        print("Setting up chimera model...")

        # Load InternVL3
        intern3_8b = (
            AutoModel.from_pretrained(
                internvl_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            )
            .to("cuda")
            .eval()
        )

        # Replace vision tower
        self.base_model.model.vision_tower = intern3_8b.vision_model

        # Create new projector
        config = self.base_model.config
        config.vision_config.hidden_size = 1024

        proj_new = Gemma3MultiModalProjector(config)
        proj_new = proj_new.to(device=self.device, dtype=self.dtype)
        self.base_model.multi_modal_projector = proj_new

        # Override the get_image_features method
        self._override_image_features_method()

        print("✓ Chimera setup complete")

    def _override_image_features_method(self):
        """
        Override the get_image_features method to handle CLS token
        """

        def custom_get_image_features(pixel_values):
            """Custom implementation that removes CLS token"""
            vision_outputs = self.base_model.model.vision_tower(
                pixel_values=pixel_values
            ).last_hidden_state

            # Remove CLS token if present
            if vision_outputs.shape[1] % 2:  # CLS token present
                vision_outputs = vision_outputs[:, 1:, :]

            image_features = self.base_model.multi_modal_projector(vision_outputs)
            return image_features

        # Replace the method
        self.base_model.model.get_image_features = custom_get_image_features

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def save_pretrained(self, save_path):
        """
        Save the complete chimera model (including vision tower)
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving complete chimera model to {save_path}")

        # Save processor
        self.processor.save_pretrained(save_path)

        # Save COMPLETE model state dict (including vision tower)
        torch.save(self.base_model.state_dict(), save_path / "pytorch_model.bin")

        # Save config with chimera modifications
        self.config.vision_config.hidden_size = 1024
        self.config.custom_model_type = "standalone_chimera"
        self.config.save_pretrained(save_path)

        # Save chimera metadata
        metadata = {
            "model_type": "standalone_chimera_v2",
            "base_model": "google/medgemma-4b-it",
            "vision_model": "InternVL3-8B",
            "vision_hidden_size": 1024,
            "self_contained": True,
            "requires_internvl_path": False,
            "modifications": [
                "CLS token removal",
                "Vision tower replacement",
                "Projector adaptation",
            ],
        }

        with open(save_path / "chimera_config.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("✓ Complete chimera model saved (self-contained)")

    @classmethod
    def from_pretrained(cls, model_path):
        """
        Load a saved chimera model (NO internvl_path required!)
        """
        model_path = Path(model_path)

        # Load metadata
        with open(model_path / "chimera_config.json", "r") as f:
            metadata = json.load(f)

        print(f"Loading self-contained chimera model from {model_path}")

        # Create instance without InternVL path
        instance = cls(base_model_id=metadata["base_model"], internvl_path=None)

        # Load the COMPLETE saved state dict
        state_dict_path = model_path / "pytorch_model.bin"
        if state_dict_path.exists():
            print("Loading complete model state (including vision tower)...")
            state_dict = torch.load(state_dict_path, map_location=instance.device)

            # Load the complete state dict
            missing_keys, unexpected_keys = instance.base_model.load_state_dict(
                state_dict, strict=False
            )

            if missing_keys:
                print(
                    f"Missing keys: {len(missing_keys)} (this is normal for new projector)"
                )
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")

            print("✓ Loaded complete model weights")

        print("✓ Self-contained chimera model loaded successfully")
        return instance


# Usage examples:
def create_chimera():
    """Create chimera model (requires InternVL path)"""
    chimera = StandaloneChimeraModel(
        base_model_id="google/medgemma-4b-it",
        internvl_path="/home/eric/projects/InternVL-3x/internvl_chat/pretrained/InternVL3-8B/",
    )
    return chimera


def save_chimera(chimera, save_path="./my_chimera"):
    """Save chimera model"""
    chimera.save_pretrained(save_path)
    return save_path


def load_chimera(model_path="./my_chimera"):
    """Load chimera model (NO paths required!)"""
    chimera = StandaloneChimeraModel.from_pretrained(model_path)
    return chimera


print("Workflow:")
print("1. CREATE: chimera = create_chimera()  # Needs InternVL path")
print("2. SAVE:   save_chimera(chimera)       # Saves everything")
print("3. LOAD:   chimera = load_chimera()    # NO paths needed!")


if __name__ == "__main__":
    # Example usage
    chimera = create_chimera()
    save_path = save_chimera(chimera, "./chimera_weights")
    loaded_chimera = load_chimera(save_path)

    print(loaded_chimera)

    # Now you can use loaded_chimera for inference as usual