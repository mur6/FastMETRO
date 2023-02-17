from pathlib import Path
import torch

from src.modeling.model import T3EncDecModel, DecWide128Model, SimpleCustomModel, OnlyRadiusModel


def save_checkpoint(model, epoch, iteration=None):
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / f"checkpoint-{epoch}"
    checkpoint_dir.mkdir(exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model

    torch.save(model_to_save, checkpoint_dir / "model.bin")
    torch.save(model_to_save.state_dict(), checkpoint_dir / "state_dict.bin")
    print(f"Save checkpoint to {checkpoint_dir}")
    return checkpoint_dir


def load_model_from_dir(resume_dir):
    if (resume_dir / "model.bin").exists() and (resume_dir / "state_dict.bin").exists():
        if torch.cuda.is_available():
            model = torch.load(resume_dir / "model.bin")
            state_dict = torch.load(resume_dir / "state_dict.bin")
        else:
            model = torch.load(resume_dir / "model.bin", map_location=torch.device("cpu"))
            state_dict = torch.load(resume_dir / "state_dict.bin", map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        return model
    else:
        raise Exception(f"{resume_dir} is not valid directory.")


def get_my_model(args, *, mymodel_resume_dir, fastmetro_model, device):
    print(f"My modele resume_dir: {mymodel_resume_dir}")

    if mymodel_resume_dir:
        mlp_for_radius = load_model_from_dir(mymodel_resume_dir)
        model = OnlyRadiusModel(fastmetro_model, mlp_for_radius=mlp_for_radius).to(device)
    else:
        model = OnlyRadiusModel(fastmetro_model).to(device)

    print(f"My model loaded: {model.__class__.__name__}")
    return model
