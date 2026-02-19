from transformers import AutoProcessor
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImagePipeline, LongCatImageEditPipeline

t2i_pipe = None
edit_pipe = None

def load_t2i_pipe():
    print("Loading text-to-image model...")
    if t2i_pipe is None:
        model_id = "meituan-longcat/LongCat-Image"
        processor = AutoProcessor.from_pretrained(model_id, subfolder="tokenizer")
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        ).to(device)
        t2i_pipe = LongCatImagePipeline.from_pretrained(
            model_id, transformer=transformer, text_processor=processor
        )
        t2i_pipe.to(device, torch.bfloat16)

    print("Text-to-image loaded.")
    return t2i_pipe

def load_edit_pipe():
    print("Loading image edit model...")
    if edit_pipe is None:
        model_id = "meituan-longcat/LongCat-Image-Edit"
        processor = AutoProcessor.from_pretrained(model_id, subfolder="tokenizer")
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        ).to(device)
        pipe = LongCatImageEditPipeline.from_pretrained(
            model_id, transformer=transformer, text_processor=processor
        )
        pipe.to(device, torch.bfloat16)
        
    print("Image edit loaded.")
    return pipe

async def edit(
    input_img,
    prompt: str,
    seed: int = 42,
):
    generator = torch.Generator(device).manual_seed(seed)
    pipe = load_edit_pipe()
    result = pipe(
        input_img,
        prompt,
        negative_prompt="",
        guidance_scale=4.5,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=generator
    )
    edited_img = result.images[0]
    buffer = io.BytesIO()
    edited_img.save(buffer, format="PNG")
    img_b64 = buffer.getvalue().hex()

    return {
        "output": img_b64,
        "status": "success",
    }

async def generate(
    prompt: str,
    seed: int = 42,
):
    generator = torch.Generator(device).manual_seed(seed)
    pipe = load_t2i_pipe()
    result = pipe(
        prompt,
        negative_prompt="",
        guidance_scale=4.5,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=generator
    )
    # Convert to bytes/png
    img = result.images[0]
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_b64 = buffer.getvalue().hex()

    return {
        "output": img_b64,
        "status": "success",
    }
