import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List, Tuple
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
def vedant_predicts(
        model : torch.nn.Module,
        class_names : List[str],
        image_path : str,
        image_size : Tuple[int, int] = (224, 224),
        transform : torchvision.transforms = None,
        device : torch.device = device,
):
    img = Image.open(image_path)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ]
        )
    model.to(device)
    model.eval()
    with torch.inference_mode():
        image_tensor = image_transform(img).unsqueeze(0).to(dim = 3)
        # transformed_image = image_tensor.expand(3,-1,-1)

        target_image_pred = model(transformed_image.to(device))
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)


    vedant = [class_names[target_image_pred_label],target_image_pred_probs.max().__format__(".3f")]
    return vedant



model = torch.load('vedant_pneumonia.pt',map_location=torch.device('cpu'))
model.eval()


vedant_predicts(model=model, class_names=['NORMAL', 'PNEUMONIA'], image_path='F:/minor2/chest_xray/test/NORMAL/IM-0003-0001.jpeg', image_size=( 224, 224), transform=None, device=device)