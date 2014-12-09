import gtkutils.img_util as iu
import numpy

def blend_plot_and_frame_fn(plot_fn, frame_fn, frame_weight = .5):
    p = iu.o(plot_fn)
    f = iu.o(frame_fn)
    return blend_plot_and_frame(p, f, frame_weight)

def blend_plot_and_frame(p, f, frame_weight = .5):
    p = iu.imresize(p, f.shape)
    
    p = (p/float(p.max())).astype('float32')

    f = (f/float(f.max())).astype('float32')

    blend = (frame_weight * f + (1 - frame_weight) * p[:, :, :3])

    blend = (blend / blend.max() * 255).astype('uint8')

    return blend

    
