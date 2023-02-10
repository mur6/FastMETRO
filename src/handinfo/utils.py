import torch


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
