from fastai.vision.all import *
import re
from torchvision import transforms
import torch


model_trash_ai = load_learner('new_restnet.pkl')

# Define the mapping from types to bin colors
bin_types = {
    "cardboard": "Blue",
    "paper": "Blue",
    "plastic": "Yellow",
    "metal": "Yellow",
    "recyclable": "Yellow",
    "glass": "Green",
    "household": "Grey",
    "compost": "Brown",
    "organic": "Brown",
    "trash": "Brown",
    "other": "Brown"
}

def predict_using_restnet_newly_trained(filename):
    try:
        prediction = model_trash_ai.predict(filename)
        num = int(prediction[1].numpy().tolist())
        prob = round(float(prediction[2].numpy()[num]),2)
        print(f'Classified as {prediction[0]}, Class number {num} with probability {prob}')
        return {'predicted': prediction[0], 'class_number': num, 'probability': prob}
    except:
        print(sys.exc_info()[0])
        return 'Error'



# Load the pre-trained restnet50 model
model_resnet50 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
model_resnet50.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_using_restnet50(filename):

    # Load and transform the image
    image = Image.open(filename)
    image = transform(image)
    image = image.unsqueeze(0)  # add batch dimension

    # Perform the prediction
    with torch.no_grad():
        output = model_resnet50(image)
        _, predicted = torch.max(output, 1)
        probabilities = F.softmax(output, dim=1)  # apply softmax to get probabilities


    with open('imagenet-simple-labels.json') as f:
        labels = json.load(f)

    label = labels[predicted.item()]

    # Get the garbage bin type for the predicted class
    bin_type = map_labels_to_bins_level1([label])[0]

    # Get the probability of the predicted class
    probability_score = probabilities[0][predicted.item()].item()
    probability_score = round(probability_score, 2)

    return {"bin_type": bin_type, "label": label, "probability_score": probability_score}

def map_labels_to_bins_level1(labels):
    # Define the mapping from labels to types
    label_to_type = {
        "plastic bag": "plastic",
        "water bottle": "plastic",
        "wine bottle": "glass",
        "beer bottle": "glass",
        "tin can": "metal",
        "pop bottle": "plastic",
        "soda bottle": "plastic",
        "packet": "paper",
        "envelope": "paper",
        "carton": "paper",
        "paper towel": "paper",
        "toilet paper": "paper",
        "newspaper": "paper",
        "notebook": "paper",
        "book jacket": "paper",
        "comic book": "paper",
        "plate": "paper",
        "menu": "paper",
        "pizza": "compost",
        "pot pie": "compost",
        "burrito": "compost",
        "hot pot": "compost",
        "trifle": "compost",
        "ice cream": "compost",
        "pretzel": "compost",
        "cheeseburger": "compost",
        "hot dog": "compost",
        "mashed potato": "compost",
        "broccoli": "compost",
        "cauliflower": "compost",
        "zucchini": "compost",
        "spaghetti squash": "compost",
        "acorn squash": "compost",
        "butternut squash": "compost",
        "cucumber": "compost",
        "artichoke": "compost",
        "bell pepper": "compost",
        "cardoon": "compost",
        "mushroom": "compost",
        "Granny Smith": "compost",
        "strawberry": "compost",
        "orange": "compost",
        "lemon": "compost",
        "fig": "compost",
        "pineapple": "compost",
        "banana": "compost",
        "jackfruit": "compost",
        "custard apple": "compost",
        "pomegranate": "compost",
        "hay": "compost",
        "carbonara": "compost",
        "chocolate syrup": "compost",
        "dough": "compost",
        "meatloaf": "compost",
        "guacamole": "compost",
        "consomme": "compost",
        "baguette": "compost",
        "bagel": "compost",
        "red wine": "compost",
        "espresso": "compost",
        "cup": "compost",
        "eggnog": "compost",
        "alp": "compost",
        "bubble": "compost",
        "cliff": "compost",
        "coral reef": "compost",
        "geyser": "compost",
        "lakeshore": "compost",
        "promontory": "compost",
        "shoal": "compost",
        "seashore": "compost",
        "valley": "compost",
        "volcano": "compost",
        "baseball player": "compost",
        "bridegroom": "compost",
        "scuba diver": "compost",
        "rapeseed": "compost",
        "daisy": "compost",
        "yellow lady's slipper": "compost",
        "corn": "compost",
        "acorn": "compost",
        "rose hip": "compost",
        "horse chestnut seed": "compost",
        "coral fungus": "compost",
        "agaric": "compost",
        "gyromitra": "compost",
        "stinkhorn mushroom": "compost",
        "earth star": "compost",
        "hen-of-the-woods": "compost",
        "bolete": "compost",
        "ear": "compost",
        "toilet paper": "compost"
    }

    # Define the mapping from types to bin colors
    bin_types = {
        "cardboard": "Blue",
        "paper": "Blue",
        "plastic": "Yellow",
        "metal": "Yellow",
        "recyclable": "Yellow",
        "glass": "Green",
        "household": "Grey",
        "compost": "Brown",
        "organic": "Brown",
        "trash": "Brown",
        "other": "Brown"
    }

    # Map the labels to types and then to bin colors
    bins = []
    for label in labels:
        if label in label_to_type:
            type_ = label_to_type[label]
            bin_color = bin_types[type_]
            bins.append(bin_color)
        else:
            bin_type_ = map_labels_to_bins_level2(label)
            bin_color = bin_types[bin_type_]
            bins.append(bin_color)
            # bins.append(bin_types["trash"])  # If the label is not recognized, it is considered as trash
    return bins

def map_labels_to_bins_level2(label):
    if label in ['bottle', 'can', 'plastic bag', 'glass bottle', 'plastic container', 'drink carton', 'metal object', 'aluminium foil', 'takeout containers', 'plastic utensils', 'styrofoam piece', 'aluminium can', 'plastic bag', 'plastic film', 'plastic tube', 'ziploc bag', 'aluminium tray', 'plastic lid', 'metal lid', 'glass jar', 'metal can']:
        return 'recyclable'
    elif label in ['paper', 'cardboard', 'paper bag', 'broken glass', 'egg carton', 'magazine', 'newspaper', 'paper cup', 'book']:
        return 'paper'
    elif label in ['food waste']:
        return 'organic'
    elif label in ['battery', 'clothing', 'shoes', 'straw', 'disposable cutlery', 'tissue', 'toothbrush', 'rubber band', 'scrap metal', 'wire', 'battery', 'broken ceramics', 'cd', 'cigarette butt', 'cooking oil', 'cork', 'electronic waste', 'light bulb', 'nail', 'paint can', 'printer cartridge', 'spectacles', 'sponge', 'styrofoam peanuts', 'tinfoil', 'tire', 'toothpaste', 'washing liquid bottle']:
        return 'household'
    else:
        return 'trash'

