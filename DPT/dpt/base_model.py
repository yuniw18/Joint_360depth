import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        if True:
            model_dict = self.state_dict()
            pretrained_dict = torch.load(path , map_location=torch.device("cpu"))
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            self.load_state_dict(model_dict)

