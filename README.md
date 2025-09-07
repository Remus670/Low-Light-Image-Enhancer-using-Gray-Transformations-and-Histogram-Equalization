# Low-Light Image Enhancer using Gray Transformations and Histogram Equalization

This project focuses on enhancing low-light grayscale images using classical image processing techniques. The application allows real-time visualization of multiple intensity transformation methods together with their histograms through an interactive graphical interface.

The project was developed using Python, OpenCV, NumPy, Tkinter and Matplotlib.

## Implemented Methods

The application includes several enhancement techniques commonly used in digital image processing:

- Linear Transformation for brightness and contrast adjustment
- Logarithmic Transformation for enhancing dark image regions
- Gamma Correction for illumination control
- Histogram Equalization (HE) for global contrast enhancement
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for adaptive local contrast improvement

Each transformation can be adjusted interactively using sliders, allowing direct comparison of the results.

## Main Functionalities

- Interactive GUI for image enhancement
- Real-time image processing and visualization
- Dynamic histogram generation
- Adjustable processing parameters
- Manual implementation of Histogram Equalization and CLAHE algorithms
- Automatic image resizing for improved performance

## Project Structure

### `Low_light.py`

Main application file containing:
- image loading
- transformation algorithms
- histogram computation
- GUI implementation
- real-time processing logic

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Tkinter

## How to Run

Install the required libraries:

```bash
pip install opencv-python numpy matplotlib