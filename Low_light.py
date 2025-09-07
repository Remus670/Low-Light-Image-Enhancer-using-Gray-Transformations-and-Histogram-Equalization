import cv2
import numpy as np
from tkinter import Tk, filedialog, Scale, HORIZONTAL, OptionMenu, StringVar, Label, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

Tk().withdraw()
file_path = filedialog.askopenfilename(
    title="Select image",
    filetypes=[("Image files", "*.jpg *.png *.bmp *.jpeg *.tif *.tiff")]
)
if not file_path:
    raise FileNotFoundError("No image selected")

image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Could not load image")

if image.shape[0] > 600 or image.shape[1] > 600:
    scale = 600 / max(image.shape)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    print(f"[INFO] Image resized to {image.shape[1]}x{image.shape[0]} for performance.")

def linear_transformation(image, a, b):
    out = a * image + b
    return np.clip(out, 0, 255).astype(np.uint8)

def log_transformation(image):
    c = 255.0 / np.log(1.0 + 255.0)
    out = c * np.log(1.0 + image.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)

def gamma_correction(image, gamma):
    img_n = image / 255.0
    out = 255.0 * (img_n ** gamma)
    return np.clip(out, 0, 255).astype(np.uint8)

def he_manual(img):
    hist = np.bincount(img.flatten(), minlength=256).astype(np.float64)

    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]
    total = img.size

    eq_map = np.round((cdf - cdf_min) / (total - cdf_min) * 255).clip(0, 255).astype(np.uint8)

    return eq_map[img]

def clahe_manual(img, clip_limit=0.02, tile_size=None):
    h, w = img.shape

    if tile_size is None:
        th = max(16, h // 8)
        tw = max(16, w // 8)
    else:
        th, tw = tile_size

    ny = max(1, (h + th - 1) // th)
    nx = max(1, (w + tw - 1) // tw)

    abs_clip = max(2, int(clip_limit * th * tw * 4))

    maps = np.zeros((ny, nx, 256), dtype=np.float32)

    for ty in range(ny):
        for tx in range(nx):
            y0, x0 = ty * th, tx * tw
            block = img[y0:min(y0 + th, h), x0:min(x0 + tw, w)]

            hist = np.bincount(block.flatten(), minlength=256).astype(np.int64)

            excess = int(np.sum(np.maximum(hist - abs_clip, 0)))
            hist = np.minimum(hist, abs_clip)
            hist += excess // 256
            hist[:excess % 256] += 1

            cdf = hist.cumsum().astype(np.float64)
            nz = cdf[cdf > 0]
            if len(nz) == 0:
                maps[ty, tx] = np.arange(256, dtype=np.float32)
                continue

            cdf_min = nz[0]
            total = cdf[-1]

            if total > cdf_min:
                maps[ty, tx] = ((cdf - cdf_min) / (total - cdf_min) * 255).clip(0, 255).astype(np.float32)
            else:
                maps[ty, tx] = np.arange(256, dtype=np.float32)

    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)

    ty_f = (ys - th / 2.0) / th
    tx_f = (xs - tw / 2.0) / tw

    ty0 = np.clip(np.floor(ty_f).astype(np.int32), 0, ny - 1)
    ty1 = np.clip(ty0 + 1, 0, ny - 1)
    tx0 = np.clip(np.floor(tx_f).astype(np.int32), 0, nx - 1)
    tx1 = np.clip(tx0 + 1, 0, nx - 1)

    wy = (ty_f - np.floor(ty_f)).astype(np.float32)[:, np.newaxis]
    wx = (tx_f - np.floor(tx_f)).astype(np.float32)[np.newaxis, :]

    v00 = maps[ty0[:, None], tx0[None, :], img]
    v01 = maps[ty0[:, None], tx1[None, :], img]
    v10 = maps[ty1[:, None], tx0[None, :], img]
    v11 = maps[ty1[:, None], tx1[None, :], img]

    out = (v00 * (1 - wy) * (1 - wx) +
           v01 * (1 - wy) * wx +
           v10 * wy * (1 - wx) +
           v11 * wy * wx)

    return np.clip(out, 0, 255).astype(np.uint8)

def compute_histogram(img):
    return np.bincount(img.flatten(), minlength=256)

root = Tk()
root.title("Interactive Processing - PNI Project")
root.configure(bg="#003366")

fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(9, 4))
plt.tight_layout()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

control_frame = Frame(root, bg="#003366")
control_frame.pack(pady=5)
Label(control_frame, text="Transformation:", bg="#003366", fg="white").grid(row=0, column=0)

transform_type = StringVar(value="Original")
options = ["Original", "Linear", "Logarithmic", "Gamma", "HE", "CLAHE"]
opt_menu = OptionMenu(control_frame, transform_type, *options)
opt_menu.config(bg="#0055aa", fg="white", activebackground="#0077cc", activeforeground="white")
opt_menu.grid(row=0, column=1)

slider_a = Scale(control_frame, from_=5, to=20, resolution=1, orient=HORIZONTAL,
                 label="a (Linear x0.1)", bg="#003366", fg="white",
                 troughcolor="#0055aa", highlightbackground="#003366")
slider_a.set(12)
slider_a.grid(row=1, column=0, columnspan=2)

slider_b = Scale(control_frame, from_=-50, to=50, resolution=1, orient=HORIZONTAL,
                 label="b (Linear)", bg="#003366", fg="white",
                 troughcolor="#0055aa", highlightbackground="#003366")
slider_b.set(15)
slider_b.grid(row=2, column=0, columnspan=2)

slider_gamma = Scale(control_frame, from_=1, to=30, resolution=1, orient=HORIZONTAL,
                     label="γ (Gamma x0.1)", bg="#003366", fg="white",
                     troughcolor="#0055aa", highlightbackground="#003366")
slider_gamma.set(8)
slider_gamma.grid(row=3, column=0, columnspan=2)

slider_clip = Scale(control_frame, from_=1, to=20, resolution=1, orient=HORIZONTAL,
                    label="CLAHE clip limit (x0.01)", bg="#003366", fg="white",
                    troughcolor="#0055aa", highlightbackground="#003366")
slider_clip.set(2)
slider_clip.grid(row=4, column=0, columnspan=2)

def update_image(*_):
    trans = transform_type.get()
    a = slider_a.get() / 10.0
    b = slider_b.get()
    gamma = slider_gamma.get() / 10.0
    clip = slider_clip.get() / 100.0

    if trans == "Linear":
        img_out = linear_transformation(image, a, b)
    elif trans == "Logarithmic":
        img_out = log_transformation(image)
    elif trans == "Gamma":
        img_out = gamma_correction(image, gamma)
    elif trans == "HE":
        img_out = he_manual(image)
    elif trans == "CLAHE":
        img_out = clahe_manual(image, clip_limit=clip)
    else:
        img_out = image.copy()

    ax_img.clear()
    ax_img.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    ax_img.set_title(f"{trans}", color='white')
    ax_img.axis('off')

    ax_hist.clear()
    hist = compute_histogram(img_out)
    ax_hist.plot(hist)
    ax_hist.set_xlim(0, 255)
    ax_hist.set_title("Histogram", color='white')
    ax_hist.grid(True, alpha=0.3)

    ax_img.set_facecolor('#003366')
    ax_hist.set_facecolor('#003366')
    fig.patch.set_facecolor('#003366')
    canvas.draw()

transform_type.trace_add("write", lambda *_: update_image())
slider_a.config(command=lambda v: update_image())
slider_b.config(command=lambda v: update_image())
slider_gamma.config(command=lambda v: update_image())
slider_clip.config(command=lambda v: update_image())

update_image()
root.mainloop()