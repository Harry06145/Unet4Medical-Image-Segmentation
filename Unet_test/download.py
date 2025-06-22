import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrewmvd/lung-vessel-segmentation")

print("Path to dataset files:", path)