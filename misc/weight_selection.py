import os
import argparse
import torch

from dust3r.inference import load_model
from dust3r.model import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
MODEL_KD = "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='dpt', \
            output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
            enc_embed_dim=384, enc_depth=12, enc_num_heads=6, dec_embed_dim=192, dec_depth=12, dec_num_heads=3, adapter=True)" # TinyDecoder

def uniform_element_selection(wt, s_shape):
    assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
    ws = wt.clone()
    for dim in range(wt.dim()):
        assert wt.shape[dim] >= s_shape[dim], "Teacher's dimension should not be smaller than student's dimension"  # determine whether teacher is larger than student on this dimension
        if wt.shape[dim] % s_shape[dim] == 0:
            step = wt.shape[dim] // s_shape[dim]
            indices = torch.arange(s_shape[dim]) * step
        else:
            indices = torch.round(torch.linspace(0, wt.shape[dim]-1, s_shape[dim])).int()
        ws = torch.index_select(ws, dim, indices)
    assert ws.shape == s_shape
    return ws

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    teacher = load_model("checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", 'cpu')
    student = eval(MODEL_KD)

    teacher_weights = teacher.state_dict()
    student_weights = student.state_dict()

    weight_selection = {}
    for key in student_weights.keys():
        if "adapter" in key:
            continue
        weight_selection[key] = uniform_element_selection(teacher_weights[key], student_weights[key].shape)

    if args.output_dir.endswith(".pt") or args.output_dir.endswith(".pth"):
        torch.save(weight_selection, os.path.join(args.output_dir))
    else:
        torch.save(weight_selection, os.path.join(args.output_dir, f"{args.model_type}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='checkpoints/', help="Output directory for saved model")
    parser.add_argument("--model_type", type=str, default='small_tiny', help="Model type: vit or convnext")

    args = parser.parse_args()
    main(args)