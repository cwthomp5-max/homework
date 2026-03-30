import gradio as gr
import numpy as np
from PIL import Image, ImageFilter
import cv2


def apply_gaussian_blur(image, radius):
    pil_img = Image.fromarray(image)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred)


def apply_lens_blur(image, radius, intensity):
    kernel_size = max(3, 2 * radius + 1)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = radius
    cv2.circle(kernel, (center, center), radius, 1, -1)
    kernel /= kernel.sum()

    img_float = image.astype(np.float32)
    blurred_channels = []
    for c in range(img_float.shape[2]):
        channel = img_float[:, :, c]
        blurred_channel = cv2.filter2D(channel, -1, kernel)
        blurred_channels.append(blurred_channel)

    blurred = np.stack(blurred_channels, axis=2)
    output = cv2.addWeighted(img_float, 1 - intensity, blurred, intensity, 0)
    return np.clip(output, 0, 255).astype(np.uint8)


def process_image(image, effect_type, gaussian_radius, lens_radius, lens_intensity):
    if image is None:
        return None, "Please upload an image first."

    img_array = np.array(image)

    if effect_type == "Gaussian Blur":
        result = apply_gaussian_blur(img_array, radius=gaussian_radius)
        label = f"Gaussian Blur — Radius: {gaussian_radius}"
    else:
        result = apply_lens_blur(img_array, radius=int(lens_radius), intensity=lens_intensity)
        label = f"Lens Blur — Radius: {lens_radius}, Intensity: {lens_intensity:.2f}"

    return Image.fromarray(result), label


with gr.Blocks(title="Blur Effects Studio") as demo:

    gr.Markdown("# 🔭 Blur Effects Studio")
    gr.Markdown("Upload an image and apply **Gaussian Blur** or **Lens (Bokeh) Blur**.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            effect_type = gr.Radio(
                choices=["Gaussian Blur", "Lens Blur (Bokeh)"],
                value="Gaussian Blur",
                label="Effect Type",
            )
            gaussian_radius = gr.Slider(
                minimum=0.5, maximum=30.0, value=5.0, step=0.5,
                label="Gaussian Radius (for Gaussian Blur)",
            )
            lens_radius = gr.Slider(
                minimum=1, maximum=25, value=8, step=1,
                label="Lens Radius (for Lens Blur)",
            )
            lens_intensity = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.8, step=0.05,
                label="Blend Intensity (for Lens Blur)",
            )
            run_btn = gr.Button("✨ Apply Effect", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Result")
            effect_label = gr.Textbox(label="Applied Effect", interactive=False)

    run_btn.click(
        fn=process_image,
        inputs=[input_image, effect_type, gaussian_radius, lens_radius, lens_intensity],
        outputs=[output_image, effect_label],
    )

    gr.Markdown("---")
    gr.Markdown(
        "**How to use:** Upload a photo → Select an effect → Adjust sliders → Click Apply Effect"
    )


if __name__ == "__main__":
    demo.launch()
