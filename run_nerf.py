import json
from PIL import Image

import numpy as np

import sys

import torch

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import debug_helpers as h
import random

class RFNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_size = 128
        self.pos_encoding_size = 27
        self.linear1 = nn.Linear(self.pos_encoding_size, self.layer_size)
        self.linear2 = nn.Linear(self.layer_size, self.layer_size)
        self.linear3 = nn.Linear(self.layer_size, self.layer_size)
        self.linear4 = nn.Linear(self.layer_size, self.layer_size)
        self.final = nn.Linear(self.layer_size, 4)
        
        
        
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        torch.nn.init.kaiming_normal_(self.linear2.weight)
        torch.nn.init.kaiming_normal_(self.linear3.weight)
        torch.nn.init.kaiming_normal_(self.linear4.weight)
        torch.nn.init.kaiming_normal_(self.final.weight)
        
        
    def forward(self, ray_position):
        x = F.relu(self.linear1(ray_position))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.final(x))
        
        sigma = F.relu(x[:, :1])
        color = F.relu(x[:, 1:])
        
        x = F.relu(sigma)
        x = F.sigmoid(color)
        
        return sigma, color

def positional_encoding(positions):
    
    encoding = [positions]
    
    n = 4
    
    bands = 2.0 ** np.linspace(0.0, n-1, n)
    
    for freq in bands:
        for func in [np.sin, np.cos]:
            encoding.append(func(positions * freq))
    
    #print(np.concatenate(encoding, axis=-1).shape)
    return np.concatenate(encoding, axis=-1)

def convert_pixels_to_ray(x, y, width, height, fov):
    ndc_x = x / width * 2.0 - 1.0
    ndc_y = -(y / height * 2.0 - 1.0)
    
    focal_length = 1.0 / np.tan(fov/2)
    
    directions = np.array((ndc_x/focal_length, ndc_y/focal_length, -np.ones_like(ndc_y)))
    
    return directions
    
def rays_to_camera(camera_transforms, camera_indices, ray_directions):
    if camera_indices != None:
        chosen_camera_transforms = camera_transforms[camera_indices]
        origins = chosen_camera_transforms[:, :3, -1]
            
        directions = np.matmul(chosen_camera_transforms[:, :3, :3], ray_directions.transpose()[:, :, np.newaxis])[:, :, 0]
    else:
        chosen_camera_transforms = camera_transforms
            
        origins = np.array([chosen_camera_transforms[:3, -1] for i in range(ray_directions.shape[1])])
        rays_to_mult = ray_directions.transpose()[:, :, np.newaxis]
        directions = np.matmul(chosen_camera_transforms[:3, :3], rays_to_mult)[:, :, 0]
    
    #directions_premult = np.multiply(chosen_camera_transforms[:, :3, :3], ray_directions.transpose())
    #directions = np.matmul(chosen_camera_transforms[:, :3, :3], ray_directions.transpose()[:, :, np.newaxis])[:, :, 0]
    

    p2 = origins + directions
    
    #h.print_obj_file(origins, p2)
    
    return origins, directions
    
    
def sample_density(model, origins, directions, near, far):
    sample_bins = 128

    range_vals = np.linspace(near, far, sample_bins+1)
    start = range_vals[:-1]
    end = range_vals[1:]
    
    samples = np.random.random_sample((sample_bins,))
    
    samples_along_ray = (end - start) * samples + start
    
    #sample_points = []
    
    density_sum = torch.zeros(len(origins))
    
    color_sum = torch.zeros((len(origins), 3))
    
    difference = samples_along_ray[0] - near
    
    for i in range(sample_bins):
    
                
        if i < sample_bins - 1:
            difference = samples_along_ray[i+1] - samples_along_ray[i]
            
        else:
            difference = 1e10
            
        current_sample_point = positional_encoding(origins + samples_along_ray[i] * directions)
        
        network_reading, color = model(torch.tensor(current_sample_point).float())
        
        cum_sum = torch.exp(-density_sum)
        curr_density = 1 - torch.exp(-network_reading[:, 0] * difference)
        
        local_density = (cum_sum * curr_density)
        
        color_sum += torch.transpose(local_density * torch.transpose(color, 0, 1), 0, 1)
            
        density_sum += network_reading[:, 0] * difference
            
        
    overall_density = 1 - torch.exp(-density_sum)
    overall_colors = color_sum
    
    return overall_density, overall_colors

def train_network(model, train, fov):
    width = 100
    height = 100
    fov = fov
    train_count = 100
    
    transforms, images = train[1], train[0]
    
    #images = torch.tensor(images).float()
    
    n_batches = 300
    batch_size = 4096
    epoch_time = 100
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    
    model.train()
    loss_values = []
    
    for i in range(n_batches):
        #print(i)
        optimizer.zero_grad()
    
    
        
        batch_image_choice = random.randint(0, train_count-1)
        print(i, batch_image_choice)
        
            
        x, y = np.linspace(0, width-1, width), np.linspace(0, height-1, height)
    
        pixel_x, pixel_y = np.meshgrid(x, y)
    
        
        target_pixels = train[0][batch_image_choice].reshape((width*height, 4))[:, :3]
        print(target_pixels.shape)
        
        
        remake_image = torch.reshape(torch.tensor(target_pixels), (width, height, 3))
        
        
        #plt.imshow(remake_image.detach().numpy())
        #plt.show()
        
        target_pixels = torch.tensor(target_pixels).float()
        
        ray_directions = convert_pixels_to_ray(pixel_x.flatten(), pixel_y.flatten(), width, height, fov)
        transformed_ray_origins, transformed_ray_directions = rays_to_camera(train[1][batch_image_choice], None, ray_directions)
        
        normalized_ray_directions = np.linalg.norm(transformed_ray_directions)
        
        density_samples, color_samples = sample_density(model, transformed_ray_origins, transformed_ray_directions, 2, 6)
        
        target = (target_pixels / 255.0)
        
        
        loss = F.mse_loss(color_samples, target[:, :3])
        
        
        remake_image = torch.reshape(torch.tensor(color_samples), (width, height, 3))
                
        #plt.imshow(remake_image.detach().numpy())
        #plt.show()
        
        
        
        loss.backward()
                
        optimizer.step()
        print(loss.detach().item())
        loss_values.append(loss.detach().item())
        if i % epoch_time == epoch_time - 1:
            scheduler.step()
            
    plt.plot(loss_values)
    plt.show()
        
        
        
        
def render_final_image(view, model, fov):
    width = 100
    height = 100
    
    x, y = np.linspace(0, width-1, width), np.linspace(0, height-1, height)
    
    pixel_x, pixel_y = np.meshgrid(x, y)
    
    ray_directions = convert_pixels_to_ray(pixel_x.flatten(), pixel_y.flatten(), width, height, fov)
    
            
    transformed_ray_origins, transformed_ray_directions = rays_to_camera(view, None, ray_directions)
            
    print(transformed_ray_origins.shape, transformed_ray_directions.shape)
            
    density_samples, color_samples = sample_density(model, transformed_ray_origins, transformed_ray_directions, 2, 6)
    
    
    final_img = torch.reshape(color_samples, (width, height, 3))
    return final_img.numpy()
    
        
        

def load_image_set(filename, count, transforms_file):
    images = []
    images_masked = []

    transforms_file_handle = open(transforms_file)
    transforms = json.load(transforms_file_handle)
    
    all_transforms = []

    fov = transforms['camera_angle_x']

    for i in range(count):
        file = filename + "r_" + str(i) + ".png"
        image = Image.open(file)
        image = image.resize((100, 100))
        converted = np.array(image)
        image_masked = converted[:, :, 3]
        
        images.append(image)
        images_masked.append(image_masked)
        
        all_transforms.append(transforms['frames'][i]['transform_matrix'])
        
    images = np.array(images)
    images_masked = np.array(images_masked)
    
    transforms_array = np.array(all_transforms)
    
    #print(transforms_array.shape)
    
    return images, transforms_array, images_masked, fov
    
        

def prepare_dataset(filename, train_count, test_count):
    
    
    train_images_directory = filename + "/train/"
    test_images_directory = filename + "/test/"
    test_transforms_filenames = filename + "/transforms_test.json"
    train_transforms_filenames = filename + "/transforms_train.json"
    
    train_images, train_transforms, train_masks, fov = load_image_set(train_images_directory, int(train_count), train_transforms_filenames)
    test_images, test_transforms, test_masks, fov = load_image_set(test_images_directory, int(test_count), test_transforms_filenames)
    
    plt.imshow(train_images[0])
    plt.show()
    
    train = (train_images, train_transforms)
    test = (test_images, test_transforms)
    
    return train, test, fov
    


if __name__ == "__main__":
    filename = sys.argv[1]
    
    train_count = sys.argv[2]
    test_count = sys.argv[3]
    
    #print(test_count, train_count)

    train, test, fov = prepare_dataset(filename, train_count, test_count)
    
    model = RFNetwork()
    
    
    train_network(model, train, fov)
    
    model.eval()
    
    
    torch.save(model.state_dict(), "model_save_path.dat")
    
    with torch.no_grad():
        for i in range(int(test_count)):

            img = render_final_image(test[1][i], model, fov)
            plt.imshow(img)
            plt.show()
            plt.imshow(test[0][i])
            plt.show()
    
    
    
