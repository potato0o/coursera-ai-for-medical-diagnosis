# Week 3 Image Segmentation on MRI Images

Summary: Using UNET for brain tumor MRI segmentation
Status: Done

## MRI Data and Labels

Medical imaging relies on specific formats for storing and sharing images due to the high complexity and need for maintaining precise information. There are some of the most commonly used formats:

### **1. DICOM (Digital Imaging and Communications in Medicine)**

- **Overview**: DICOM is the most widely used format in medical imaging. It standardizes the handling, storing, printing, and transmission of medical images and related data.
- **Features**:
    - Includes a header with patient information and image metadata.
    - Supports a variety of imaging modalities, including X-rays, MRIs, CT scans, and ultrasounds.
    - Ensures interoperability between different devices and systems.
- **Usage**: Used universally in hospitals, clinics, and diagnostic centers.

### **2. NIfTI (Neuroimaging Informatics Technology Initiative)**

- **Overview**: NIfTI is commonly used in the neuroimaging community. It was developed to improve upon the Analyze format, another neuroimaging data format.
- **Features**:
    - Supports single-file storage with extensions like .nii or dual-file storage with .hdr and .img files.
    - Designed for flexibility and ease of use in neuroimaging research.
    - Provides a straightforward way to store 3D and 4D imaging data.
- **Usage**: Primarily used in brain imaging studies and research.

### **3. NRRD (Nearly Raw Raster Data)**

- **Overview**: NRRD is designed to store and share raster data and is particularly suited for scientific imaging.
- **Features**:
    - Supports various data types and dimensionalities.
    - Simple format for storing arrays of raw data.
    - Easy to integrate with scientific visualization tools.
- **Usage**: Used in medical imaging research and computational modeling.

### **4. Analyze**

- **Overview**: An older format commonly used in brain imaging before NIfTI.
- **Features**:
    - Uses two files: one for header information (.hdr) and one for image data (.img).
    - Supports a range of data types and dimensions.
- **Usage**: Less common today but still used in legacy systems and older research data.

### **5. MINC (Medical Imaging NetCDF)**

- **Overview**: Developed by the Montreal Neurological Institute, MINC is used for multidimensional medical imaging data.
- **Features**:
    - Highly flexible and extensible.
    - Supports multiple dimensions and complex metadata.
    - Built on the NetCDF data format, facilitating large dataset management.
- **Usage**: Primarily in neurological research and for handling large imaging datasets.

### **6. JPEG2000**

- **Overview**: An image compression standard and coding system. Unlike the standard JPEG, JPEG2000 offers lossless and lossy compression.
- **Features**:
    - Superior compression techniques result in smaller file sizes with high image quality.
    - Supports large image sizes and multiple resolutions.
    - Can be embedded in DICOM files for improved compression and storage efficiency.
- **Usage**: Used in radiology and other fields where high-quality image storage and transmission are necessary.

Each of these formats serves different needs within medical imaging, from standardized communication and interoperability (DICOM) to specialized research applications (NIfTI, MINC). The choice of format depends on the specific requirements of the medical application, including the type of imaging, the need for metadata, and the intended use of the images.

## DICOM (Digital Imaging and Communication in Medicine)

DICOM (Digital Imaging and Communications in Medicine) is a comprehensive standard used in medical imaging to ensure the interoperability of systems that produce, store, retrieve, and transmit medical images. A DICOM file includes not just the image data but also a wealth of information about the image, patient, and equipment. Here's a detailed look at the kind of information included in a DICOM file:

### 1. **Header Information**

The DICOM header contains metadata that is crucial for interpreting the image data correctly. This includes:

- **Patient Information**:
    - Patient's name
    - Patient's ID
    - Date of birth
    - Gender
    - Other identifiers such as medical record numbers
- **Study Information**:
    - Study ID
    - Study date and time
    - Referring physician’s name
    - Accession number (a unique identifier for the imaging study)
- **Series Information**:
    - Series number
    - Modality (e.g., CT, MRI, Ultrasound)
    - Series description
- **Image Information**:
    - Instance number (unique identifier for each image within a series)
    - Image date and time
    - Image orientation and position
    - Pixel spacing and dimensions
    - Image resolution
- **Equipment Information**:
    - Manufacturer of the imaging device
    - Model name
    - Device settings and parameters used during image acquisition (e.g., X-ray tube current, voltage, slice thickness for CT)

### 2. **Image Data**

- The actual pixel data representing the image. This can be a single frame or multiple frames (as in the case of CT or MRI scans which generate a series of images).

### 3. **Annotations and Overlays**

- Annotations such as regions of interest (ROI), text notes, or graphic overlays can be included within the DICOM file to highlight specific areas of interest on the image.

### 4. **Additional Metadata**

- Information about the image acquisition process, such as scan protocols, contrast agents used, or details about patient positioning.
- Details about the software versions and the digital systems used to process and store the images.

### DICOM File Structure

A DICOM file is typically divided into several parts:

- **Preamble**: A 128-byte section that is usually empty but can be used for compatibility with non-DICOM systems.
- **DICOM Prefix**: The letters "DICM" to indicate that the file adheres to the DICOM standard.
- **Data Elements**: The bulk of the file is composed of data elements, each with a tag (identifier), a length, and a value. Data elements can contain patient details, image attributes, and the image pixel data itself.

### Benefits of DICOM

- **Interoperability**: Ensures that medical images and associated data can be exchanged between different systems and devices from various manufacturers.
- **Integration**: Facilitates integration with other healthcare systems, such as PACS (Picture Archiving and Communication Systems), EHR (Electronic Health Records), and radiology information systems.
- **Standardization**: Promotes consistency and standardization in how medical images are stored, retrieved, and displayed, enhancing the reliability and quality of patient care.

### Conclusion

DICOM is a robust and versatile standard that encapsulates not only the image data but also extensive information about the patient, imaging procedure, and equipment. This ensures comprehensive documentation and facilitates the accurate interpretation and sharing of medical imaging data across different platforms and institutions.

## NIFTI

NIfTI (Neuroimaging Informatics Technology Initiative) is a file format widely used in neuroimaging for storing and sharing complex brain imaging data. It was designed to address some of the limitations of its predecessor, the Analyze format, and to improve the ease of use, flexibility, and standardization in the neuroimaging community. Here is detailed information about NIfTI:

### Overview of NIfTI

NIfTI is primarily used to store MRI, fMRI, PET, and other brain imaging data. It supports both single-file (.nii) and dual-file formats (.hdr and .img), making it versatile and convenient for a variety of research and clinical applications.

### File Structure

NIfTI files can be in one of two formats:

1. **Single-File Format (.nii)**
    - Combines the header and image data into one file.
    - Simplifies file management and reduces the risk of losing part of the data.
2. **Dual-File Format (.hdr/.img)**
    - The header information is stored in a .hdr file.
    - The image data is stored in a .img file.
    - This separation can sometimes be advantageous for certain processing workflows.

### Components of a NIfTI File

A NIfTI file contains several key pieces of information:

1. **Header Information**
    - **Data Type**: Specifies the type of data stored (e.g., integer, float).
    - **Dimensions**: Describes the size of the image (number of voxels in each dimension).
    - **Voxel Dimensions**: Specifies the size of each voxel in real-world units (e.g., millimeters).
    - **Coordinate System**: Information about the orientation of the image in space (e.g., which direction is up, which is left).
    - **Transformation Matrices**: Matrices used to convert voxel indices to real-world coordinates, ensuring accurate spatial orientation.
    - **Auxiliary Information**: Can include details such as slice timing for fMRI data, intent codes for statistical maps, and other metadata.
2. **Image Data**
    - **Voxel Data**: The core data of the file, representing the actual intensity values at each voxel. This can be 3D (for a static scan) or 4D (for dynamic scans like fMRI).

### Advantages of NIfTI

1. **Simplified Data Management**: The single-file format (.nii) makes it easier to handle and transfer files without losing associated metadata.
2. **Rich Metadata**: The header includes comprehensive information about the image, aiding in accurate interpretation and analysis.
3. **Compatibility**: NIfTI is supported by many neuroimaging software packages, including SPM, FSL, and AFNI, ensuring broad compatibility and ease of use.
4. **Standardization**: By providing a consistent format, NIfTI helps standardize neuroimaging data storage, making it easier to share and compare datasets across studies.

### Usage

- **MRI and fMRI Studies**: Commonly used for storing structural and functional MRI data due to its ability to handle complex 3D and 4D datasets.
- **Brain Mapping**: Facilitates the creation and sharing of brain maps, such as activation maps from fMRI studies or statistical maps from group analyses.
- **Data Sharing**: Widely adopted in neuroimaging research communities, promoting data sharing and collaborative studies.

### Conclusion

NIfTI is a powerful and flexible file format tailored for the needs of the neuroimaging community. Its ability to handle rich metadata, combined with ease of use and broad software support, makes it an essential tool for researchers and clinicians working with brain imaging data.

## Use DICOM and NIFTI File in PyTorch

Importing and using DICOM and NIfTI formats in PyTorch involves several steps, including reading the image files, preprocessing the data, and converting it into tensors that can be used for training models. Here’s a guide to handle both formats:

### Using DICOM with PyTorch

1. **Install Required Libraries**
You will need libraries like `pydicom` for reading DICOM files and `torch` for PyTorch functionalities.
    
    ```bash
    pip install pydicom torch torchvision
    ```
    
2. **Reading DICOM Files**
Use `pydicom` to read DICOM files and convert them into NumPy arrays, which can then be converted into PyTorch tensors.
    
    ```python
    import pydicom
    import numpy as np
    import torch
    import torchvision.transforms as transforms
    
    # Read the DICOM file
    dicom_file = pydicom.dcmread('path_to_dicom_file.dcm')
    
    # Extract the pixel array
    pixel_array = dicom_file.pixel_array
    
    # Normalize the pixel values (optional, depends on your use case)
    pixel_array = pixel_array.astype(np.float32) / np.max(pixel_array)
    
    # Convert the NumPy array to a PyTorch tensor
    tensor = torch.tensor(pixel_array)
    
    # Add a channel dimension if needed (e.g., for grayscale images)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    
    print(tensor.shape)  # For example: torch.Size([1, 512, 512])
    ```
    
3. **Preprocessing**
You can use `torchvision.transforms` for preprocessing like resizing, normalization, and augmentation.
    
    ```python
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Apply the transform to the tensor
    transformed_tensor = transform(tensor)
    
    print(transformed_tensor.shape)
    ```
    

### Using NIfTI with PyTorch

1. **Install Required Libraries**
You will need `nibabel` for reading NIfTI files and `torch` for PyTorch functionalities.
    
    ```bash
    pip install nibabel torch torchvision
    ```
    
2. **Reading NIfTI Files**
Use `nibabel` to read NIfTI files and convert them into NumPy arrays, which can then be converted into PyTorch tensors.
    
    ```python
    import nibabel as nib
    import numpy as np
    import torch
    import torchvision.transforms as transforms
    
    # Load the NIfTI file
    nifti_file = nib.load('path_to_nifti_file.nii')
    
    # Get the image data as a NumPy array
    image_data = nifti_file.get_fdata()
    
    # Normalize the image data (optional)
    image_data = image_data.astype(np.float32) / np.max(image_data)
    
    # Convert the NumPy array to a PyTorch tensor
    tensor = torch.tensor(image_data)
    
    # Add a channel dimension if needed (e.g., for grayscale images)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    
    print(tensor.shape)  # For example: torch.Size([1, 64, 64, 64])
    ```
    
3. **Preprocessing**
Similar to DICOM, you can use `torchvision.transforms` for preprocessing.
    
    ```python
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Apply the transform to the tensor
    transformed_tensor = transform(tensor)
    
    print(transformed_tensor.shape)
    ```
    

### Example: Putting It All Together

Here is a full example for both formats:

### DICOM Example

```python
import pydicom
import torch
import torchvision.transforms as transforms

def load_dicom_as_tensor(dicom_path):
    dicom_file = pydicom.dcmread(dicom_path)
    pixel_array = dicom_file.pixel_array
    pixel_array = pixel_array.astype('float32') / np.max(pixel_array)
    tensor = torch.tensor(pixel_array).unsqueeze(0)
    return tensor

dicom_path = 'path_to_dicom_file.dcm'
dicom_tensor = load_dicom_as_tensor(dicom_path)
print(dicom_tensor.shape)
```

### NIfTI Example

```python
import nibabel as nib
import torch

def load_nifti_as_tensor(nifti_path):
    nifti_file = nib.load(nifti_path)
    image_data = nifti_file.get_fdata()
    image_data = image_data.astype('float32') / np.max(image_data)
    tensor = torch.tensor(image_data).unsqueeze(0)
    return tensor

nifti_path = 'path_to_nifti_file.nii'
nifti_tensor = load_nifti_as_tensor(nifti_path)
print(nifti_tensor.shape)
```

These examples demonstrate how to read medical images from DICOM and NIfTI formats, convert them into tensors, and optionally preprocess them for use in PyTorch. Adjust the paths and preprocessing steps according to your specific use case and data requirements.

## Image Registration

Image registration is a crucial process in medical imaging and other fields, where it is necessary to align two or more images of the same scene taken at different times, from different viewpoints, or by different sensors. The goal is to transform different sets of data into one coordinate system so that they can be compared or analyzed together. Here’s a detailed explanation of the concepts, techniques, and practical steps for image registration:

### Key Concepts

1. **Fixed Image (Reference)**: The image that remains unchanged during the registration process.
2. **Moving Image**: The image that is transformed to align with the fixed image.
3. **Transformation**: The mathematical operations used to align the moving image with the fixed image. Transformations can be rigid (translation and rotation) or non-rigid (elastic deformations).
4. **Interpolation**: A method to estimate pixel values at non-grid positions after transformation.
5. **Similarity Metric**: A measure used to quantify how well the moving image aligns with the fixed image. Common metrics include Mean Squared Error (MSE), Mutual Information (MI), and Cross-Correlation.

### Techniques

1. **Rigid Registration**: Involves translation and rotation. It is useful when the objects in the images are assumed to be rigid bodies.
2. **Affine Registration**: Includes scaling, shearing, translation, and rotation. It is more flexible than rigid registration.
3. **Non-Rigid (Elastic) Registration**: Handles complex deformations where parts of the image may move differently. It’s essential for aligning soft tissues in medical imaging.
4. **Feature-Based Registration**: Uses distinct features (e.g., corners, edges) to align images. Common algorithms include SIFT, SURF, and ORB.
5. **Intensity-Based Registration**: Uses pixel intensity values directly and optimizes similarity metrics. Techniques like Mutual Information are popular here.

### Practical Steps for Image Registration Using PyTorch

### Step 1: Install Required Libraries

You need libraries like OpenCV for image processing and PyTorch for deep learning functionalities.

```bash
pip install opencv-python torch torchvision
```

### Step 2: Load Images

```python
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

# Load fixed and moving images
fixed_image = cv2.imread('fixed_image_path.png', cv2.IMREAD_GRAYSCALE)
moving_image = cv2.imread('moving_image_path.png', cv2.IMREAD_GRAYSCALE)

# Convert images to float32
fixed_image = fixed_image.astype(np.float32)
moving_image = moving_image.astype(np.float32)
```

### Step 3: Define the Similarity Metric

For this example, we’ll use Mean Squared Error (MSE) as the similarity metric.

```python
def mse_loss(fixed, moving):
    return torch.mean((fixed - moving) ** 2)
```

### Step 4: Apply Transformation

We will apply a simple translation as an example of a rigid transformation. For more complex transformations, optimization techniques such as gradient descent or evolutionary algorithms can be used.

```python
def translate_image(image, tx, ty):
    rows, cols = image.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, M, (cols, rows))
    return translated_image

# Example transformation parameters
tx, ty = 10, 15  # Translate 10 pixels right and 15 pixels down

# Apply transformation
transformed_image = translate_image(moving_image, tx, ty)

# Convert to tensors
fixed_tensor = torch.tensor(fixed_image).unsqueeze(0).unsqueeze(0)
moving_tensor = torch.tensor(transformed_image).unsqueeze(0).unsqueeze(0)
```

### Step 5: Optimization

A basic optimization loop to minimize the similarity metric.

```python
import torch.optim as optim

# Initialize transformation parameters
tx, ty = torch.tensor([0.0], requires_grad=True), torch.tensor([0.0], requires_grad=True)

# Define optimizer
optimizer = optim.SGD([tx, ty], lr=1.0)

# Optimization loop
num_iterations = 100
for i in range(num_iterations):
    optimizer.zero_grad()

    # Apply transformation
    transformed_image = translate_image(moving_image, tx.item(), ty.item())
    moving_tensor = torch.tensor(transformed_image).unsqueeze(0).unsqueeze(0)

    # Calculate loss
    loss = mse_loss(fixed_tensor, moving_tensor)

    # Backpropagate
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss.item()}, tx: {tx.item()}, ty: {ty.item()}")
```

### Step 6: Result Visualization

```python
import matplotlib.pyplot as plt

# Apply final transformation
final_transformed_image = translate_image(moving_image, tx.item(), ty.item())

# Display the images
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Fixed Image')
plt.imshow(fixed_image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Moving Image')
plt.imshow(moving_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Aligned Image')
plt.imshow(final_transformed_image, cmap='gray')
plt.show()
```

### Advanced Registration Techniques

For more complex registration tasks, consider using deep learning approaches such as convolutional neural networks (CNNs) or advanced optimization algorithms. Libraries such as SimpleITK and Elastix provide comprehensive tools for more sophisticated image registration workflows.

By following these steps and techniques, you can perform image registration in PyTorch, aligning medical images or any other types of images for further analysis and processing.

## Sub-sectioning of 3D Image

Sub-sectioning a 3D image involves dividing a larger 3D volume into smaller, more manageable sections or patches. This technique is particularly useful in medical imaging for various applications, including deep learning, where smaller input sizes are required, and it enhances the analysis of specific regions of interest within the 3D image.

### **Why Sub-Section 3D Images?**

1. **Memory Management**: Large 3D images can be too big to fit into memory all at once, especially when working with high-resolution scans.
2. **Data Augmentation**: Creating more training samples by extracting overlapping patches.
3. **Uniform Input Size**: Neural networks often require fixed-size inputs.
4. **Localized Analysis**: Focus on specific regions within the 3D volume for detailed analysis.

### **Steps to Sub-Section 3D Images**

1. **Load the 3D Image**: Use libraries like **`nibabel`** for NIfTI files or **`pydicom`** for DICOM files.
2. **Define Patch Size and Stride**: Determine the size of each patch and the step size for moving the patching window.
3. **Extract Patches**: Implement a function to extract patches from the 3D image.
4. **Convert to PyTorch Tensors**: Convert the extracted patches to tensors for use in PyTorch.
5. **Data Augmentation (Optional)**: Apply augmentation techniques to increase the diversity of your dataset.
