import torch


def xy_location_to_num(input_size, crop_size, in_x, in_y,out_n_regions):
    out_n_regions = np.sqrt(out_n_regions)
    out_stride = int((input_size - crop_size) / (out_n_regions - 1))


    out_x, out_y = round(in_x / out_stride), round(in_y / out_stride)
    return int(out_x * out_n_regions + out_y)

def get_patch_predictions(image, patch_size, model, num_classes, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    # Make an attribution map, by sliding the predictor over the target image
    image = image.clone()
    image = image.to(device)
    patch_size=32
    # with stride one, no padding, go over every possible patch
    
    patches = []
    locations = []
    for i in range(0,image.shape[1]-patch_size + 2,2):
        for j in range(0,image.shape[2]-patch_size + 2,2):
            # extract patch and interpolate location
            patch = image[:,j:j+patch_size,i:i+patch_size].clone()
            location = xy_location_to_num(image.shape[1], patch_size, i,j, 2401)
    
            patches.append(patch)
            locations.append(location)

    # divide the image into patches
    patches = torch.stack(patches).reshape((-1,49,3,32,32))
    locations = torch.tensor(locations).reshape((-1,49))
    locations= locations.to(device)

    # loop over all mini batches and do predictions
    all_predictions = []
    for i in range(patches.shape[0]):
        with torch.no_grad():
            predictions = model(patches[i], locations[i])
            all_predictions.append(predictions)
    preds = torch.stack(all_predictions).reshape(-1,num_classes)

    return preds