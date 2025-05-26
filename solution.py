# Standard library imports
import traceback # For more detailed error messages if needed

# Third-party library imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# --- Task 1: Dataset Generation Placeholder ---
def generate_3d_coule_dataset(num_volumes: int = 5, base_size_h: int = 64, base_size_w: int = 64, base_size_d: int = 32) -> list[np.ndarray]:
    """
    Placeholder for Task 1: Generating a 3D version of the Coule dataset.
    As per README: "Estendere il data set sintetico 2D denominato Coule (fornito) al 3D."
    
    This function currently generates random dummy 3D volumes. In a real implementation,
    it would create structured phantoms based on the Coule dataset principles.

    Args:
        num_volumes (int): Number of dummy 3D volumes to generate.
        base_size_h (int): Base height for the volumes. Chosen to be divisible by 8 
                           for compatibility with the ResUNet3D model's pooling layers.
        base_size_w (int): Base width for the volumes, similar to height.
        base_size_d (int): Base original depth for the volumes. Sub-volumes (slices) 
                           will be extracted from this depth for network processing.

    Returns:
        list[np.ndarray]: A list of dummy 3D NumPy arrays, each of shape (H, W, D_original).
                          Data is float32 and randomly valued in [0,1].
    """
    print(f"Placeholder Function: Generating {num_volumes} dummy 3D 'Coule' dataset volumes...")
    dataset = []
    for i in range(num_volumes):
        # Introduce some variability in generated volume sizes, while maintaining compatibility.
        # Dimensions are perturbed by multiples of 8 to keep them divisible by 8.
        h = base_size_h + np.random.randint(-2, 3) * 8 
        w = base_size_w + np.random.randint(-2, 3) * 8
        # Depth variation is also by multiples of 8, but can be smaller overall.
        d = base_size_d + np.random.randint(-2, 3) * 4 
        
        # Ensure minimum dimensions after randomization
        h = max(32, h) # Minimum H, W: 32 (e.g., for 3 pooling layers 32 -> 16 -> 8 -> 4)
        w = max(32, w)
        d = max(16, d) # Minimum original depth must be at least nf_slices for network.
                       # This ensures enough depth to extract sub-volumes.
        
        dataset.append(np.random.rand(h, w, d).astype(np.float32))
    print(f"  Generated {len(dataset)} volumes. Example shape: {dataset[0].shape if dataset else 'N/A'}")
    return dataset

# --- Utility for Sub-volume Extraction ---
def extract_sub_volumes(volume: np.ndarray, nf_slices: int) -> np.ndarray:
    """
    Extracts overlapping sub-volumes (stacks of slices) from a given 3D volume.
    The input volume is assumed to be in (Height, Width, Original_Depth) format.
    The output sub-volumes are formatted as (Num_Sub_Volumes, Slices_Per_SubVolume, Height, Width),
    which is suitable for batch processing by a 3D CNN where 'Slices_Per_SubVolume' acts as the depth.

    Args:
        volume (np.ndarray): A single 3D NumPy array (H, W, D_original).
        nf_slices (int): The number of adjacent slices to stack for each sub-volume.
                         This defines the 'depth' dimension for the network input.

    Returns:
        np.ndarray: A 4D NumPy array of shape (Num_Sub_Volumes, nf_slices, H, W).
                    Returns an empty array if `nf_slices` is too large for the volume's depth
                    or if `nf_slices` is not positive.
    """
    if not isinstance(volume, np.ndarray):
        raise TypeError("Input 'volume' must be a NumPy array.")
    if volume.ndim != 3:
        raise ValueError(f"Input 'volume' must be 3-dimensional (H, W, D), got {volume.ndim} dimensions.")
    if not isinstance(nf_slices, int) or nf_slices <= 0:
        raise ValueError("'nf_slices' must be a positive integer.")

    original_h, original_w, original_depth = volume.shape

    if nf_slices > original_depth:
        # This situation might occur if a fixed nf_slices is used for varied dataset volumes.
        # print(f"Warning: nf_slices ({nf_slices}) is greater than the volume depth ({original_depth}). Returning empty array.")
        return np.array([]) # No valid sub-volumes can be extracted.

    # Calculate the number of sub-volumes that can be extracted.
    # This is a sliding window approach along the depth axis.
    num_sub_volumes = original_depth - nf_slices + 1
    
    sub_volumes_list = []
    for i in range(num_sub_volumes):
        # Extract a chunk of `nf_slices` depth: shape (H, W, nf_slices)
        sub_volume_chunk = volume[:, :, i : i + nf_slices]
        # Transpose to (nf_slices, H, W) to match typical PyTorch 3D CNN input format (D, H, W for a single sub-volume)
        sub_volume_transposed = np.transpose(sub_volume_chunk, (2, 0, 1))
        sub_volumes_list.append(sub_volume_transposed)
        
    if not sub_volumes_list:
        return np.array([]) # Should not happen if num_sub_volumes > 0.

    return np.stack(sub_volumes_list, axis=0)

# --- Evaluation Metric Functions ---
def calculate_re(output: np.ndarray, target: np.ndarray) -> float:
    """
    Calculates the Relative Error (RE) between the output and target arrays.
    RE = ||output - target||_2 / ||target||_2, where ||.||_2 is the L2 norm (Frobenius norm).

    Args:
        output (np.ndarray or torch.Tensor): The predicted data. If Tensor, converted to NumPy.
        target (np.ndarray or torch.Tensor): The ground truth data. If Tensor, converted to NumPy.

    Returns:
        float: The calculated Relative Error. Returns np.inf if the L2 norm of the target is zero.
    """
    if isinstance(output, torch.Tensor): 
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor): 
        target = target.detach().cpu().numpy()

    diff_norm = np.linalg.norm(output - target)
    target_norm = np.linalg.norm(target)
    
    if target_norm == 0:
        # Avoid division by zero. If target_norm is 0, RE is undefined or infinite
        # unless output_norm is also 0 (in which case RE could be 0).
        return np.inf if diff_norm > 1e-9 else 0.0 
    return diff_norm / target_norm

def calculate_psnr(output: np.ndarray, target: np.ndarray, data_range: float) -> float:
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between the output and target arrays.
    Utilizes skimage.metrics.peak_signal_noise_ratio.

    Args:
        output (np.ndarray or torch.Tensor): The predicted data. If Tensor, converted to NumPy.
        target (np.ndarray or torch.Tensor): The ground truth data. If Tensor, converted to NumPy.
        data_range (float): The range of the data (e.g., max_val - min_val). 
                            For normalized data [0,1], this is 1.0.

    Returns:
        float: The PSNR value in decibels (dB).
    """
    if isinstance(output, torch.Tensor): 
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor): 
        target = target.detach().cpu().numpy()
        
    return psnr_metric(target, output, data_range=data_range)

def calculate_ssim(output: np.ndarray, target: np.ndarray, data_range: float) -> float:
    """
    Calculates the Structural Similarity Index (SSIM) between the output and target arrays.
    Utilizes skimage.metrics.structural_similarity. 
    Handles 3D volumes by averaging SSIM over slices or direct 3D comparison if supported.

    Args:
        output (np.ndarray or torch.Tensor): The predicted data. If Tensor, converted to NumPy.
        target (np.ndarray or torch.Tensor): The ground truth data. If Tensor, converted to NumPy.
        data_range (float): The range of the data. For normalized data [0,1], this is 1.0.

    Returns:
        float: The SSIM value, typically between -1 and 1 (higher is better).
    """
    if isinstance(output, torch.Tensor): 
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor): 
        target = target.detach().cpu().numpy()

    # Ensure inputs are 3D (D, H, W) for consistent SSIM calculation.
    # If input is (Batch=1, Channels=1, D, H, W), squeeze out Batch and Channel dimensions.
    if output.ndim == 5 and output.shape[0] == 1 and output.shape[1] == 1: # B,C,D,H,W
        output = output.squeeze(0).squeeze(0) 
    if target.ndim == 5 and target.shape[0] == 1 and target.shape[1] == 1: # B,C,D,H,W
        target = target.squeeze(0).squeeze(0)
    # If input is (Channels=1, D, H, W), squeeze out Channel dimension.
    if output.ndim == 4 and output.shape[0] == 1: 
        output = output.squeeze(0)
    if target.ndim == 4 and target.shape[0] == 1: 
        target = target.squeeze(0)

    if target.ndim != 3:
        # print(f"SSIM Warning: Target shape {target.shape} is not 3D (D,H,W). Returning NaN.")
        # SSIM in skimage typically expects 2D images or 3D volumes.
        # For N-D arrays, it might try to compute it slice-wise if channel_axis is not specified.
        return np.nan 

    # Ensure win_size for SSIM is appropriate: odd, <= smallest dimension, and >= 3.
    min_dim = min(target.shape)
    win_size = min(7, min_dim) # Default win_size is 7, but cap at min_dim
    if win_size < 3 : win_size = min_dim # If min_dim is very small (e.g. 1 or 2), use it.
    if win_size % 2 == 0: win_size -= 1 # Ensure it's odd
    win_size = max(3, win_size) # Must be at least 3 for meaningful local stats.
    
    # For 3D volumes (D,H,W), skimage's ssim computes it slice by slice along the first axis (depth)
    # by default if channel_axis is None and data_format is not explicitly set for 3D.
    # We can also specify data_format='DDD' or 'DHW' once available, or use channel_axis for multichannel 2D.
    # Current skimage version might average SSIM over slices for 3D input.
    return ssim_metric(target, output, data_range=data_range, win_size=win_size, channel_axis=None)


# --- 3D ResUNet Architecture ---
class ConvBlock3D(nn.Module):
    """
    A basic 3D Convolutional Block: Conv3D -> BatchNorm3D -> ReLU.
    This block is a fundamental component of the ResUNet architecture.

    Args:
        in_channels (int): Number of input channels to the 3D convolution.
        out_channels (int): Number of output channels from the 3D convolution.
        kernel_size (int, optional): Size of the convolving kernel. Defaults to 3.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding added to all six sides of the input. Defaults to 1.
                                 Padding=1 for kernel_size=3 ensures same output spatial/depth dimensions.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # BatchNorm bias is typically False when followed by ReLU, but default is True. Conv bias is False as BN has affine params.
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True) # Inplace ReLU saves memory.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvBlock3D.
        Input tensor shape: (Batch, Channels_in, Depth, Height, Width)
        Output tensor shape: (Batch, Channels_out, Depth', Height', Width')
                             (D',H',W' same as D,H,W if stride=1, padding=1, kernel=3)
        """
        return self.relu(self.bn(self.conv(x)))

class ResBlock3D(nn.Module):
    """
    A 3D Residual Block, forming the core of the ResUNet encoder and decoder stages.
    It consists of two ConvBlock3D layers and a skip connection (residual connection).
    If input and output channels differ, a 1x1x1 convolution is used in the skip connection
    to match dimensions.

    Architecture:
        x_in -> ConvBlock3D_1 (in_ch -> out_ch) -> ConvBlock3D_2 (out_ch -> out_ch) -> out
          \                                                                           /
           ---------------------> ResidualConnection (1x1x1 Conv if needed) --------> + -> ReLU -> x_out

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # First convolutional block, changes channels from in_channels to out_channels
        self.conv1 = ConvBlock3D(in_channels, out_channels)
        # Second convolutional block, maintains out_channels
        self.conv2 = ConvBlock3D(out_channels, out_channels)
        
        # Residual connection: if in_channels and out_channels are different,
        # use a 1x1x1 convolution to match the dimensions. Otherwise, use identity.
        if in_channels == out_channels:
            self.residual_connection = nn.Identity()
        else:
            # This 1x1x1 conv ensures the residual can be added to the main path output.
            self.residual_connection = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            # Optional: Add BatchNorm to the residual connection as well for stability
            # self.residual_connection = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            #     nn.BatchNorm3d(out_channels)
            # )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResBlock3D.
        Input tensor shape: (Batch, C_in, D, H, W)
        Output tensor shape: (Batch, C_out, D, H, W) (assuming convs maintain D,H,W)
        """
        residual = self.residual_connection(x)  # Prepare residual
        out = self.conv1(x)                     # First conv block
        out = self.conv2(out)                   # Second conv block
        out = out + residual                    # Add residual
        return self.relu(out)                   # Final activation

class ResUNet3D(nn.Module):
    """
    3D Residual U-Net Architecture.
    This network is designed for volumetric image processing tasks, such as segmentation or reconstruction.
    It uses ResBlock3D for its encoder and decoder stages.

    The architecture consists of:
    1. Initial Convolution: An initial ConvBlock3D to extract low-level features.
    2. Encoder Path: Three stages, each with a ResBlock3D followed by MaxPool3d for downsampling.
                     Filter counts increase at each stage (e.g., f -> 2f -> 4f).
    3. Bottleneck: A ResBlock3D at the deepest part of the U-Net, with the highest filter count (e.g., 8f).
    4. Decoder Path: Three stages, each with ConvTranspose3d for upsampling, concatenation with
                     corresponding encoder stage's output (skip connection), and a ResBlock3D.
                     Filter counts decrease at each stage (e.g., 8f -> 4f -> 2f -> f).
    5. Output Layer: A final 1x1x1 Conv3d to map features to the desired number of output channels.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale volumes).
        out_channels (int): Number of output channels (e.g., 1 for reconstructed volume or num_classes for segmentation).
        init_filter_size (int, optional): Number of filters in the first convolutional layer.
                                          Subsequent layers scale this size. Defaults to 32.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_filter_size: int = 32):
        super().__init__()
        
        f = init_filter_size # Base filter count

        # --- Encoder Path ---
        # Initial convolution block
        self.enc_initial_conv = ConvBlock3D(in_channels, f) # (B, C_in, D, H, W) -> (B, f, D, H, W)
        
        # Stage 1
        self.enc_res1 = ResBlock3D(f, f)                   # (B, f, D, H, W) -> (B, f, D, H, W)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2) # (B, f, D/2, H/2, W/2)

        # Stage 2
        self.enc_res2 = ResBlock3D(f, f * 2)               # (B, f, D/2, H/2, W/2) -> (B, f*2, D/2, H/2, W/2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2) # (B, f*2, D/4, H/4, W/4)

        # Stage 3
        self.enc_res3 = ResBlock3D(f * 2, f * 4)           # (B, f*2, D/4, H/4, W/4) -> (B, f*4, D/4, H/4, W/4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2) # (B, f*4, D/8, H/8, W/8)

        # --- Bottleneck ---
        self.bottleneck = ResBlock3D(f * 4, f * 8)         # (B, f*4, D/8, H/8, W/8) -> (B, f*8, D/8, H/8, W/8)

        # --- Decoder Path ---
        # Stage 3 (Up)
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2) # Upsample: (B, f*8, D/8,H/8,W/8) -> (B, f*4, D/4,H/4,W/4)
        # Input to dec_res3 is concatenation of upsampled output and encoder skip connection (f*4 + f*4 = f*8 channels)
        self.dec_res3 = ResBlock3D(f * 4 + f * 4, f * 4)                      # Output: (B, f*4, D/4, H/4, W/4)

        # Stage 2 (Up)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2) # Upsample: (B, f*4, D/4,H/4,W/4) -> (B, f*2, D/2,H/2,W/2)
        self.dec_res2 = ResBlock3D(f * 2 + f * 2, f * 2)                      # Output: (B, f*2, D/2, H/2, W/2)

        # Stage 1 (Up)
        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)     # Upsample: (B, f*2, D/2,H/2,W/2) -> (B, f, D,H,W)
        self.dec_res1 = ResBlock3D(f + f, f)                                  # Output: (B, f, D, H, W)

        # --- Output Layer ---
        # 1x1x1 Convolution to map features to the desired number of output channels.
        self.out_conv = nn.Conv3d(f, out_channels, kernel_size=1, stride=1) # (B, f, D,H,W) -> (B, C_out, D,H,W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResUNet3D.
        Input tensor shape: (Batch, in_channels, Depth, Height, Width)
        Output tensor shape: (Batch, out_channels, Depth, Height, Width) 
                             (Assumes all convs/pooling are configured to maintain or predictably 
                              alter D,H,W, and upsampling restores them for the output layer)
        """
        # Encoder Path
        s0 = self.enc_initial_conv(x) # Output of initial conv, used as skip for dec_res1 if needed (currently using s1)
        s1 = self.enc_res1(s0)        # Output of first ResBlock stage (skip connection for d1)
        p1 = self.pool1(s1)           # Pooled output
        
        s2 = self.enc_res2(p1)        # Output of second ResBlock stage (skip connection for d2)
        p2 = self.pool2(s2)
        
        s3 = self.enc_res3(p2)        # Output of third ResBlock stage (skip connection for d3)
        p3 = self.pool3(s3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder Path with Skip Connections
        d3_up = self.up3(b)                # Upsample bottleneck output
        d3_cat = torch.cat([d3_up, s3], dim=1) # Concatenate with skip connection from enc_res3 (s3)
        d3 = self.dec_res3(d3_cat)         # Pass through ResBlock
        
        d2_up = self.up2(d3)
        d2_cat = torch.cat([d2_up, s2], dim=1) # Concatenate with s2
        d2 = self.dec_res2(d2_cat)
        
        d1_up = self.up1(d2)
        d1_cat = torch.cat([d1_up, s1], dim=1) # Concatenate with s1 (output of first encoder ResBlock)
        d1 = self.dec_res1(d1_cat)

        # Output Layer
        out = self.out_conv(d1)
        # Activation function (e.g., Sigmoid for [0,1] output, or none for regression) depends on task.
        # For this generic model, no final activation is applied here.
        return out

# --- Training Function ---
def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, num_epochs: int = 3, device: str = 'cpu'):
    """
    Trains a PyTorch model using the provided data, loss function, and optimizer.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
                                   Expected to yield (inputs, targets) tuples.
        criterion (nn.Module): The loss function (e.g., nn.MSELoss, nn.CrossEntropyLoss).
        optimizer (optim.Optimizer): The optimization algorithm (e.g., Adam, SGD).
        num_epochs (int, optional): Number of epochs to train for. Defaults to 3.
        device (str, optional): The device to train on ('cpu' or 'cuda'). Defaults to 'cpu'.
    """
    model.to(device)  # Move model to the specified device
    model.train()     # Set the model to training mode
    
    print(f"  Starting training on {device} for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device) # Move batch data to device
            
            # Standard training steps:
            optimizer.zero_grad()      # 1. Zero the parameter gradients
            outputs = model(inputs)    # 2. Forward pass: compute predicted outputs
            loss = criterion(outputs, targets) # 3. Calculate the loss
            loss.backward()            # 4. Backward pass: compute gradient of the loss w.r.t. model parameters
            optimizer.step()           # 5. Perform a single optimization step (parameter update)
            
            running_loss += loss.item() # Accumulate loss for the epoch
            
            # Print progress periodically
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader): # Print every 5 batches or last batch
                print(f'    Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Batch Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        print(f'  Epoch [{epoch+1}/{num_epochs}] completed. Average Training Loss: {epoch_loss:.4f}')
    print("  Training finished.")

# --- Placeholder for Synthetic Projection Generation ---
def generate_synthetic_projections(volume: np.ndarray, geometry_params: dict) -> np.ndarray:
    """
    Placeholder for generating synthetic X-ray projections (sinogram) from a 3D volume.
    In a real implementation, this would use a ray tracing engine (e.g., ASTRA toolbox's
    projection capabilities if it supports direct volume to sinogram without file I/O,
    or another library like TIGRE, CONRAD, or custom CUDA code).

    Args:
        volume (np.ndarray): The 3D ground truth volume (H, W, D).
        geometry_params (dict): Dictionary containing parameters defining the projection geometry,
                                such as source/detector positions, number of angles, angle range, etc.
                                Example: `{'name': 'G1', 'angles': np.linspace(0, np.pi, 180)}`

    Returns:
        np.ndarray: A dummy 2D sinogram (Num_Angles x Num_Detectors_Per_Angle).
                    Values are random floats.
    """
    print(f"Placeholder Function: Generating synthetic projections for volume shape {volume.shape} with geometry: {geometry_params.get('name', 'UnnamedGeom')}")
    
    # Guess some sinogram dimensions based on input volume and geometry parameters
    num_angles = len(geometry_params.get("angles", np.linspace(0, np.pi, 180))) # Default to 180 angles
    # num_detectors typically relates to the width/height of the volume
    num_detectors = max(volume.shape[0], volume.shape[1]) 
    
    dummy_sinogram = np.random.rand(num_angles, num_detectors).astype(np.float32)
    print(f"  Generated dummy sinogram of shape: {dummy_sinogram.shape}")
    return dummy_sinogram

# --- Task 2: ASTRA Reconstruction Placeholder ---
def reconstruct_with_astra(synthetic_projections: np.ndarray, geometry_params: dict, original_vol_shape: tuple) -> np.ndarray:
    """
    Placeholder for Task 2: ASTRA-based Feldkamp-Davis-Kress (FDK) reconstruction.
    As per README: "Utilizzare il codice fornito basato su ASTRA che ricostruisce il volume
    a partire da proiezioni sintetiche."
    
    This function would typically interface with the ASTRA Tomography Toolbox to perform
    cone-beam CT reconstruction using an algorithm like FDK.

    Args:
        synthetic_projections (np.ndarray): The sinogram data (Num_Angles x Num_Detectors).
        geometry_params (dict): Dictionary containing parameters for the reconstruction geometry,
                                matching those used for projection.
        original_vol_shape (tuple): Target shape (H, W, D) for the reconstructed volume.
                                    This is used here to create a dummy volume of correct size.

    Returns:
        np.ndarray: A dummy reconstructed 3D volume of shape `original_vol_shape`.
                    Values are random floats, simulating a noisy reconstruction.
    """
    print(f"Placeholder Function: Reconstructing with 'ASTRA' for projections shape {synthetic_projections.shape} with geometry: {geometry_params.get('name', 'UnnamedGeom')}")
    
    # Simulate a reconstruction: create a random volume of the original shape
    # This might be less "perfect" than the ground truth, so scale values and add noise.
    reconstruction = np.random.rand(*original_vol_shape).astype(np.float32) * 0.8 # Simulate some signal loss/blur
    reconstruction += np.random.normal(0, 0.1, reconstruction.shape).astype(np.float32) # Add some Gaussian noise
    reconstruction = np.clip(reconstruction, 0, 1) # Ensure data range is maintained (e.g. [0,1])
    
    print(f"  Generated dummy 'ASTRA' reconstruction of shape: {reconstruction.shape}")
    return reconstruction


# --- Utility for ResUNet Processing of a Full Volume (Placeholder for Recombination) ---
def process_volume_with_resunet(volume_to_process: np.ndarray, model: ResUNet3D, 
                                nf_slices_network: int, device: str, batch_size_sub_volumes: int = 4) -> np.ndarray:
    """
    Processes a full 3D volume using a pre-trained ResUNet3D model.
    This function handles:
    1. Extracting overlapping sub-volumes if `nf_slices_network` is less than the volume's depth.
    2. Running inference with the ResUNet model on these sub-volumes.
    3. (Placeholder) Recombining the processed sub-volumes into a single output volume.

    Args:
        volume_to_process (np.ndarray): The input 3D volume (H, W, D_original).
        model (ResUNet3D): The trained PyTorch ResUNet3D model.
        nf_slices_network (int): The number of slices expected by the network as its depth dimension.
        device (str): PyTorch device ('cpu' or 'cuda').
        batch_size_sub_volumes (int): Batch size for processing sub-volumes through the model.

    Returns:
        np.ndarray: The ResUNet processed 3D volume, with the same (H, W, D_original) shape.
                    Currently, the recombination logic is a placeholder.
    """
    print(f"Processing volume of shape {volume_to_process.shape} with ResUNet (network nf_slices={nf_slices_network})...")
    model.eval().to(device) # Ensure model is in evaluation mode and on the correct device
    
    original_h, original_w, original_d = volume_to_process.shape

    # Case 1: The network's expected depth (nf_slices_network) is the entire volume's depth.
    # Process the whole volume as a single batch item.
    if nf_slices_network >= original_d:
        print("  Processing entire volume as one chunk (nf_slices_network >= volume_depth).")
        # Prepare for model: transpose to (nf_slices, H, W), add Batch and Channel dims -> (1, 1, D, H, W)
        vol_transposed = np.transpose(volume_to_process, (2, 0, 1)) # (D, H, W)
        vol_tensor = torch.from_numpy(vol_transposed).float().unsqueeze(0).unsqueeze(0).to(device) # (1,1,D,H,W)
        
        with torch.no_grad():
            processed_tensor = model(vol_tensor) # Output: (1,1,D,H,W)
        
        # Convert back to original NumPy format (H, W, D)
        processed_volume_np = processed_tensor.squeeze(0).squeeze(0).cpu().numpy() # (D,H,W)
        return np.transpose(processed_volume_np, (1, 2, 0)) # (H,W,D)

    # Case 2: Extract sub-volumes, process, and then recombine.
    # This is the more common case for large volumes and fixed-depth networks.
    print("  Extracting sub-volumes for ResUNet processing...")
    # sub_volumes_np shape: (Num_Sub, nf_slices_network, H, W)
    sub_volumes_np = extract_sub_volumes(volume_to_process, nf_slices_network)

    if sub_volumes_np.size == 0:
        print("  Warning: No sub-volumes extracted (nf_slices_network might be too large or volume too shallow). Returning original volume.")
        return volume_to_process.copy() # Or handle error appropriately

    # Prepare sub-volumes for PyTorch: add channel dimension, create DataLoader
    # Input tensor shape: (Num_Sub, 1, nf_slices_network, H, W)
    sub_volumes_tensor = torch.from_numpy(sub_volumes_np).float().unsqueeze(1) 
    sub_volumes_dataset = TensorDataset(sub_volumes_tensor) # Dataset of inputs only
    sub_volumes_loader = DataLoader(sub_volumes_dataset, batch_size=batch_size_sub_volumes, shuffle=False) # No shuffle for recombination

    print(f"  Processing {sub_volumes_np.shape[0]} sub-volumes in batches...")
    processed_sub_volumes_list = []
    with torch.no_grad():
        for batch_data in sub_volumes_loader:
            inputs_batch = batch_data[0].to(device) # DataLoader yields a list containing the tensor
            outputs_batch = model(inputs_batch)     # Model output: (batch_sz, 1, nf_slices_network, H, W)
            processed_sub_volumes_list.append(outputs_batch.cpu().numpy())
    
    # Concatenate processed batches back into a single array of sub-volumes
    # Shape: (Num_Sub, 1, nf_slices_network, H, W)
    all_processed_sub_volumes = np.concatenate(processed_sub_volumes_list, axis=0)
    # Squeeze out the channel dimension: (Num_Sub, nf_slices_network, H, W)
    all_processed_sub_volumes = all_processed_sub_volumes.squeeze(1)


    # --- Placeholder for Recombination Logic ---
    print("  NOTE: Recombination of processed sub-volumes is currently a PLACEHOLDER.")
    # A proper recombination would average overlapping regions to create a smooth final volume.
    # Example steps for proper recombination:
    # 1. Initialize `final_reconstructed_volume = np.zeros_like(volume_to_process, dtype=np.float32)`
    # 2. Initialize `overlap_counts = np.zeros_like(volume_to_process, dtype=np.int16)`
    # 3. Iterate `k` from `0` to `num_sub_volumes - 1`:
    #    `current_processed_sub_volume = all_processed_sub_volumes[k]` -> shape (nf_slices, H, W)
    #    `sub_vol_HWD = np.transpose(current_processed_sub_volume, (1, 2, 0))` -> shape (H, W, nf_slices)
    #    `start_slice_index = k`
    #    `end_slice_index = k + nf_slices_network`
    #    `final_reconstructed_volume[:, :, start_slice_index:end_slice_index] += sub_vol_HWD`
    #    `overlap_counts[:, :, start_slice_index:end_slice_index] += 1`
    # 4. `final_reconstructed_volume /= np.maximum(overlap_counts, 1)` # Avoid division by zero
    
    # Current placeholder: Naively use the processed sub-volumes to fill the original volume.
    # This will have block artifacts and only use the last sub-volumes that fit.
    # For simplicity, we'll just reconstruct the first part of the volume using the first few sub-volumes
    # or, even simpler, return a modified copy of the original as a clear placeholder.
    
    # Simplest Placeholder: Return a slightly modified copy of the input.
    # This clearly indicates that the complex recombination is not yet implemented.
    placeholder_output_volume = volume_to_process.copy() * 0.95 # Simulate some processing
    placeholder_output_volume += np.random.normal(0, 0.01, placeholder_output_volume.shape).astype(np.float32)
    placeholder_output_volume = np.clip(placeholder_output_volume, 0, 1)
    print("  Placeholder recombination: Returning a slightly modified copy of the input volume.")
    return placeholder_output_volume


# --- Function for Running Experiments (Simulates "Output Attesi") ---
def run_experiments(dataset_gt: list[np.ndarray], model_to_test: ResUNet3D, 
                    nf_slices_for_network: int, device: str):
    """
    Placeholder for running experiments as described in "Output Attesi" of the README.
    This function simulates:
    - Iterating through different acquisition geometries and noise levels.
    - Generating synthetic projections from ground truth volumes.
    - Adding noise to projections.
    - Reconstructing volumes using a placeholder ASTRA/FDK function.
    - Post-processing FDK reconstructions with the provided ResUNet model.
    - Calculating and printing RE, PSNR, SSIM metrics for FDK vs. GT and ResUNet vs. GT.
    
    The README mentions: "testato per differenti geometrie di acquisizione e differenti livelli di rumore",
    "mostrando le ricostruzioni e uno o due ritagli significativi", and 
    "calcolando la media delle metriche indicate e riportandole in una tabella."
    These latter parts (visualization, averaging, tables) are not implemented here.

    Args:
        dataset_gt (list[np.ndarray]): List of ground truth 3D volumes.
        model_to_test (ResUNet3D): The (conceptually) trained ResUNet model.
        nf_slices_for_network (int): The depth (number of slices) the ResUNet model expects.
        device (str): PyTorch device ('cpu' or 'cuda').
    """
    print("\n--- Starting Experiments (Placeholder Implementation) ---")
    
    # Define example acquisition geometries (as per README structure)
    # These would correspond to different ASTRA configurations.
    geometries = [
        {"name": "G1_LimitedAngle_11views", "angles": np.linspace(-15, 15, 11).tolist(), "notes": "Limited angle, 11 views"}, # Degrees
        {"name": "G2_SparseAngle_11views", "angles": np.linspace(-45, 45, 11).tolist(), "notes": "Sparse angle, 11 views"}, # Degrees
        {"name": "G3_MoreComplete_21views", "angles": np.linspace(-60, 60, 21).tolist(), "notes": "More complete, 21 views"} # Degrees
    ]
    noise_levels_sigma = [0.0, 0.01, 0.05] # Example standard deviations for additive Gaussian noise
    
    # Assuming data is normalized to [0,1] for consistent metric calculation
    data_range_for_metrics = 1.0 

    # Use a subset of the dataset for these example experiments (e.g., the first volume)
    # In a real study, you would iterate through all relevant volumes.
    test_volumes_gt = dataset_gt[:1] 

    for vol_idx, ground_truth_volume in enumerate(test_volumes_gt):
        print(f"\nProcessing Test Ground Truth Volume {vol_idx+1}/{len(test_volumes_gt)}, Shape: {ground_truth_volume.shape}")

        for geom_params in geometries:
            for noise_sigma in noise_levels_sigma:
                print(f"\n  Experiment Config: Geometry='{geom_params['name']}', Noise Sigma={noise_sigma:.3f}")

                # Step 1: Generate Synthetic Projections from Ground Truth
                sinogram_clean = generate_synthetic_projections(ground_truth_volume, geom_params)

                # Step 2: Add Noise to Projections
                noisy_sinogram = sinogram_clean
                if noise_sigma > 0:
                    noise = np.random.normal(0, noise_sigma, sinogram_clean.shape).astype(np.float32)
                    noisy_sinogram = sinogram_clean + noise
                    # Basic clipping to maintain approximate data range (can be more sophisticated)
                    noisy_sinogram = np.clip(noisy_sinogram, 0, np.max(sinogram_clean) if np.max(sinogram_clean) > 0 else 1.0) 

                # Step 3: Reconstruct with ASTRA (FDK) Placeholder
                # The reconstructed volume should have the same dimensions as the ground_truth_volume.
                fdk_reconstruction = reconstruct_with_astra(noisy_sinogram, geom_params, ground_truth_volume.shape)

                # Step 4: Post-process FDK Reconstruction with ResUNet
                # The `process_volume_with_resunet` function handles sub-volume extraction & recombination (currently placeholder).
                resunet_processed_volume = process_volume_with_resunet(
                    fdk_reconstruction, model_to_test, nf_slices_for_network, device
                )

                # Step 5: Calculate and Print Metrics
                print("    Metrics (Ground Truth vs. FDK Reconstruction):")
                re_fdk = calculate_re(fdk_reconstruction, ground_truth_volume)
                psnr_fdk = calculate_psnr(fdk_reconstruction, ground_truth_volume, data_range_for_metrics)
                ssim_fdk = calculate_ssim(fdk_reconstruction, ground_truth_volume, data_range_for_metrics)
                print(f"      RE: {re_fdk:.4f}, PSNR: {psnr_fdk:.2f} dB, SSIM: {ssim_fdk:.4f}")

                print("    Metrics (Ground Truth vs. ResUNet-Processed FDK):")
                re_resunet = calculate_re(resunet_processed_volume, ground_truth_volume)
                psnr_resunet = calculate_psnr(resunet_processed_volume, ground_truth_volume, data_range_for_metrics)
                ssim_resunet = calculate_ssim(resunet_processed_volume, ground_truth_volume, data_range_for_metrics)
                print(f"      RE: {re_resunet:.4f}, PSNR: {psnr_resunet:.2f} dB, SSIM: {ssim_resunet:.4f}")

    print("\nExperiments Placeholder Run Finished.")
    print("  Next steps in a real scenario would include:")
    print("  - Implementing actual Coule dataset generation, ASTRA projection/reconstruction.")
    print("  - Implementing robust sub-volume recombination in 'process_volume_with_resunet'.")
    print("  - Storing metrics systematically for table generation.")
    print("  - Generating visualizations (reconstructions, difference maps, specific ROIs/slices).")


# --- Self-contained Test for Network and Original Metric Calculation Logic ---
def test_network_and_metrics(device: str, nf_slices_test: int = 16, 
                             img_h_test: int = 64, img_w_test: int = 64,
                             init_filters_test: int = 16): # Reduced filters for speed
    """
    Encapsulates a self-contained test for the ResUNet3D model and metric calculation.
    This function performs:
    1. Generation of dummy 3D volume data.
    2. Extraction of sub-volumes for network input and target.
    3. Instantiation of the ResUNet3D model, loss function, and optimizer.
    4. A brief dummy training loop.
    5. Evaluation on a sample from the dummy data, calculating RE, PSNR, and SSIM.
    
    This helps verify that the network components and basic training/evaluation pipeline are functional.

    Args:
        device (str): PyTorch device ('cpu' or 'cuda').
        nf_slices_test (int): Number of slices for sub-volumes in this test (network depth).
        img_h_test (int): Height of the dummy volumes for this test.
        img_w_test (int): Width of the dummy volumes for this test.
        init_filters_test (int): Initial filter size for ResUNet in this test (can be smaller for speed).
    """
    print("\n--- Running Self-Contained Network and Metrics Test ---")
    
    # --- Test Data Preparation ---
    # Original depth of the raw volume from which sub-volumes are extracted
    raw_vol_depth_test = nf_slices_test + 10 # Ensure enough depth to extract some sub-volumes
    raw_volume_test = np.random.rand(img_h_test, img_w_test, raw_vol_depth_test).astype(np.float32)
    print(f"  Generated test raw volume of shape: {raw_volume_test.shape}")

    # Extract sub-volumes (input for network)
    # Shape: (Num_Sub, nf_slices_test, H, W)
    sub_volumes_inputs_np = extract_sub_volumes(raw_volume_test, nf_slices_test)
    if sub_volumes_inputs_np.size == 0:
        print("  Error: Failed to extract sub-volumes in test_network_and_metrics. Skipping test.")
        return

    # Create dummy targets (e.g., slightly perturbed inputs)
    sub_volumes_targets_np = np.clip(sub_volumes_inputs_np + 0.1 * np.random.rand(*sub_volumes_inputs_np.shape).astype(np.float32), 0, 1)
    
    # Convert to PyTorch Tensors and add Channel dimension: (Num_Sub, 1, nf_slices_test, H, W)
    inputs_tensor = torch.from_numpy(sub_volumes_inputs_np).float().unsqueeze(1)
    targets_tensor = torch.from_numpy(sub_volumes_targets_np).float().unsqueeze(1)
    print(f"  Prepared test input tensor: {inputs_tensor.shape}, Test target tensor: {targets_tensor.shape}")

    # Create DataLoader for the test
    # Batch size: use at least 1, or half the number of sub-volumes if more are available.
    test_batch_size = max(1, inputs_tensor.shape[0] // 2 if inputs_tensor.shape[0] > 1 else 1)
    test_dataset = TensorDataset(inputs_tensor, targets_tensor)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    # --- Test Model, Criterion, Optimizer ---
    # Use a smaller init_filter_size for faster testing if specified
    model_test = ResUNet3D(in_channels=1, out_channels=1, init_filter_size=init_filters_test).to(device)
    criterion_test = nn.MSELoss()
    optimizer_test = optim.Adam(model_test.parameters(), lr=0.001)
    print(f"  Test ResUNet3D model (init_filters={init_filters_test}), MSELoss, and Adam optimizer initialized on {device}.")

    # --- Brief Dummy Training ---
    print("  Starting dummy training for self-contained test (1 epoch)...")
    train_model(model_test, test_loader, criterion_test, optimizer_test, num_epochs=1, device=device)

    # --- Evaluation on a Sample ---
    print("  Evaluating on a sample for self-contained test...")
    model_test.eval() # Set model to evaluation mode
    
    # Get a sample input and target (first item from the non-shuffled tensors)
    sample_input_torch = inputs_tensor[0:1].to(device) # Shape: (1, 1, nf_slices_test, H, W)
    # Corresponding target, already in NumPy format (Num_Sub, nf_slices_test, H, W)
    sample_target_np = sub_volumes_targets_np[0]      # Shape: (nf_slices_test, H, W)
    
    with torch.no_grad(): # Disable gradient calculations for inference
        sample_output_tensor = model_test(sample_input_torch) # Output: (1, 1, nf_slices_test, H, W)
    
    # Convert output to NumPy array for metric calculation, squeeze Batch and Channel dimensions
    output_np = sample_output_tensor.squeeze(0).squeeze(0).cpu().numpy() # Shape: (nf_slices_test, H, W)

    print(f"  Sample output shape for metrics: {output_np.shape}, target shape: {sample_target_np.shape}")
    data_range_metrics = 1.0 # Assuming data normalized to [0,1]

    re_val = calculate_re(output_np, sample_target_np)
    psnr_val = calculate_psnr(output_np, sample_target_np, data_range_metrics)
    ssim_val = calculate_ssim(output_np, sample_target_np, data_range_metrics) # SSIM expects (D,H,W)
    
    print(f"  Test Metrics - RE: {re_val:.4f}, PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
    print("--- Self-Contained Network and Metrics Test Finished ---")


# --- Main Execution Logic ---
if __name__ == '__main__':
    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Main Script Start --- Using device: {device} ---")

    # --- Global Parameters for the Simulation ---
    # Depth of sub-volumes fed into the ResUNet (network's 'D' dimension)
    NF_SLICES_FOR_RESUNET = 16 
    # Base dimensions for generated dummy volumes (Height, Width)
    # Should be divisible by 8 due to 3 pooling layers in ResUNet (2*2*2=8)
    DUMMY_VOL_H = 64
    DUMMY_VOL_W = 64
    # Original depth of the generated dummy volumes. Must be >= NF_SLICES_FOR_RESUNET.
    # Adding a margin allows for multiple sub-volumes to be extracted.
    DUMMY_VOL_D_ORIGINAL = NF_SLICES_FOR_RESUNET + 10 # e.g., 16 + 10 = 26

    # --- PHASE 1: Dataset Generation (Task 1 Placeholder) ---
    print("\n--- PHASE 1: Dataset Generation (Placeholder) ---")
    # This simulates the generation of the 3D Coule dataset.
    ground_truth_dataset = generate_3d_coule_dataset(
        num_volumes=3, # Generate a small number of volumes for this example script
        base_size_h=DUMMY_VOL_H, 
        base_size_w=DUMMY_VOL_W, 
        base_size_d=DUMMY_VOL_D_ORIGINAL
    )
    
    # --- PHASE 2: Model Instantiation ---
    print("\n--- PHASE 2: ResUNet Model Instantiation ---")
    # Instantiate the ResUNet3D model that will be conceptually trained and used in experiments.
    # init_filter_size can be adjusted based on desired model capacity and memory constraints.
    resunet_model = ResUNet3D(in_channels=1, out_channels=1, init_filter_size=32).to(device)
    print(f"  ResUNet3D model instantiated with init_filter_size=32 on {device}.")

    # --- PHASE 3: Conceptual Model Training ---
    print("\n--- PHASE 3: Conceptual Model Training (Simplified Example) ---")
    # This phase simulates training the ResUNet model.
    # In a real scenario, this would involve:
    # - A much larger dataset of (FDK_reconstruction, GroundTruth_volume) pairs.
    # - Splitting data into training, validation, and test sets.
    # - More extensive data augmentation.
    # - A more rigorous training loop with validation checks, learning rate scheduling, etc.
    
    if ground_truth_dataset: # Proceed if dataset generation was successful
        # Use one volume from the dataset for this conceptual training example
        training_gt_volume = ground_truth_dataset[0] 
        print(f"  Using Ground Truth Volume of shape {training_gt_volume.shape} for conceptual training.")

        # Define example geometry and noise parameters for generating training data
        training_geometry_params = {"name": "TrainingGeom_Moderate", "angles": np.linspace(-30, 30, 31).tolist()}
        training_noise_sigma = 0.02 # Moderate noise level

        # Simulate generation of training data: GT -> Projections -> Noisy Projections -> FDK Reconstruction
        print("    Simulating generation of training sample (GT -> Projections -> FDK)...")
        train_projections_clean = generate_synthetic_projections(training_gt_volume, training_geometry_params)
        train_projections_noisy = train_projections_clean + np.random.normal(0, training_noise_sigma, train_projections_clean.shape).astype(np.float32)
        train_projections_noisy = np.clip(train_projections_noisy, 0, np.max(train_projections_clean) if np.max(train_projections_clean) > 0 else 1.0)
        
        # FDK reconstruction serves as input to the ResUNet
        train_fdk_input_volume = reconstruct_with_astra(train_projections_noisy, training_geometry_params, training_gt_volume.shape)
        print(f"    Generated FDK input volume for training, shape: {train_fdk_input_volume.shape}")

        # Prepare data for ResUNet:
        # - Input: Sub-volumes extracted from `train_fdk_input_volume`.
        # - Target: Corresponding sub-volumes extracted from `training_gt_volume`.
        input_sub_volumes_np = extract_sub_volumes(train_fdk_input_volume, NF_SLICES_FOR_RESUNET)
        target_sub_volumes_np = extract_sub_volumes(training_gt_volume, NF_SLICES_FOR_RESUNET)

        # Ensure sub-volumes were successfully extracted and shapes match
        if input_sub_volumes_np.size > 0 and target_sub_volumes_np.size > 0 and \
           input_sub_volumes_np.shape == target_sub_volumes_np.shape:
            
            print(f"    Extracted sub-volumes for training. Input shape: {input_sub_volumes_np.shape}, Target shape: {target_sub_volumes_np.shape}")
            
            # Convert to PyTorch Tensors, add Channel dimension (N, C, D, H, W), and move to device
            train_input_tensor = torch.from_numpy(input_sub_volumes_np).float().unsqueeze(1)
            train_target_tensor = torch.from_numpy(target_sub_volumes_np).float().unsqueeze(1)

            # Create DataLoader for the conceptual training
            # Batch size should be appropriate for the number of samples and GPU memory
            num_train_samples = train_input_tensor.shape[0]
            conceptual_batch_size = min(4, num_train_samples) if num_train_samples > 0 else 0
            
            if conceptual_batch_size > 0:
                conceptual_train_dataset = TensorDataset(train_input_tensor, train_target_tensor)
                conceptual_train_loader = DataLoader(conceptual_train_dataset, batch_size=conceptual_batch_size, shuffle=True)

                # Define loss function and optimizer for this training session
                criterion = nn.MSELoss() # Mean Squared Error is common for reconstruction tasks
                optimizer = optim.Adam(resunet_model.parameters(), lr=0.0005) # Adam optimizer with a learning rate

                print("    Starting conceptual training run with ResUNet...")
                # Train for a few epochs as a demonstration
                train_model(resunet_model, conceptual_train_loader, criterion, optimizer, num_epochs=2, device=device) 
                print("    Conceptual training run finished.")
            else:
                print("    Warning: Not enough sub-volumes to create DataLoader for conceptual training. Skipping.")
        else:
            print("    Error: Could not extract sub-volumes for conceptual training, or shapes mismatch.")
            print(f"      FDK input shape: {train_fdk_input_volume.shape}, GT volume shape: {training_gt_volume.shape}, nf_slices: {NF_SLICES_FOR_RESUNET}")
            print(f"      Input sub-volumes found: {input_sub_volumes_np.size > 0}, Target sub-volumes found: {target_sub_volumes_np.size > 0}")
            if input_sub_volumes_np.size > 0 and target_sub_volumes_np.size > 0:
                 print(f"      Input sub-volume shape: {input_sub_volumes_np.shape}, Target sub-volume shape: {target_sub_volumes_np.shape}")

    else: # If ground_truth_dataset is empty
        print("  No dataset generated, skipping conceptual model training.")

    # --- PHASE 4: Self-Contained Network Test (Sanity Check) ---
    # This runs an independent test of the ResUNet architecture and training pipeline
    # using its own quickly generated dummy data. Good for verifying components.
    # Reduce filter size for faster test run.
    print("\n--- PHASE 4: Running Self-Contained Network Test (Sanity Check) ---")
    test_network_and_metrics(device, nf_slices_test=NF_SLICES_FOR_RESUNET, 
                             img_h_test=DUMMY_VOL_H, img_w_test=DUMMY_VOL_W,
                             init_filters_test=16) # Use fewer filters for a quicker test

    # --- PHASE 5: Conceptual Evaluation / Running Experiments ---
    print("\n--- PHASE 5: Conceptual Evaluation / Running Experiments (Placeholder) ---")
    # This simulates the "Output Attesi" part of the README, using the (conceptually) trained model.
    if ground_truth_dataset:
        run_experiments(ground_truth_dataset, resunet_model, NF_SLICES_FOR_RESUNET, device)
    else:
        print("  No dataset generated, skipping experiments.")

    print("\n--- Main Script End ---")
