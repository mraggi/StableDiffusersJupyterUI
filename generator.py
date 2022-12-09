import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
to_pil = tv.transforms.ToPILImage()
to_tensor = tv.transforms.ToTensor()
from math import floor, ceil, sqrt
from PIL import Image, ImageDraw, ImageFont
import textwrap
from fastprogress import progress_bar, master_bar
from unified_pipeline import UnifiedStableDiffusionPipeline
from inpaint_pipeline import UnifiedInpaintPipeline
import random
import os
import pathlib
from pathlib import Path

import numpy as np

import gradio as gr
from IPython.display import display
from delegation import delegates
from math import prod
import os

def separate_by_sizes(imgs):
    imgs.sort(key=lambda x: 10000*x.width + x.height)
    R,r = [],[]
    prev_size = (imgs[0].height, imgs[0].width)
    for i in imgs:
        curr_size = (i.height, i.width)
        if curr_size != prev_size:
            R.append(r)
            r = []
            prev_size = curr_size
        r.append(i)
    R.append(r)
    
    return R

def smart_image_grid(imgs, rows=None, cols=None):
    imgs_by_sizes = separate_by_sizes(imgs)
    for I in imgs_by_sizes:
        display(image_grid(I,rows,cols))

def image_grid(imgs, rows=None, cols=None):
    bs = 30
    if (len(imgs) > bs):
        print(f"There are {len(imgs)} images. Breaking up in batches")
        blocks = range(0,len(imgs),bs)
        for b in blocks:
            c = min(b + bs,len(imgs))
            display(image_grid(imgs[b:c]))
        return
    num = len(imgs)
    if rows is None and cols is None:
        if num <= 4: rows,cols = 1,num
        else:
            cols = 3
            rows = ceil(num/cols)
    if rows is None: rows = ceil(num/cols)
    if cols is None: cols = ceil(num/rows)
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def random_string(n):
    X = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join([random.choice(X) for _ in range(n)])

def mask_from_alpha(img):
    if isinstance(img, str) or isinstance(img, Path): img = Image.open(img)
    a = to_tensor(img)
    assert(a.shape[0] == 4)
    mask = a[3:]
    return to_pil((mask < 0.8).float())

def _with_mean_std(X, mean=0., std=1.):
    mu, sigma = X.mean(), X.std()
    return (X-mu)*std/sigma + mean

class EnhancedGenerator:
    def __init__(self, pipe, inpaint_pipe=None, height=512, width=832, steps=90, guidance_scale=7.5, savedir = "saved"):
        self.pipe = pipe
        self.device = pipe.device
        self.inpaint_pipe = pipe if inpaint_pipe is None else inpaint_pipe
        self.height = height
        self.width = width
        self.steps = steps
        self.guidance_scale = guidance_scale
        
        self.saved = []
        
        self.savedir = Path(savedir)
        (self.savedir/"pth").mkdir(exist_ok=True, parents=True)
        self.show_as_generated = False   
        
    @delegates(UnifiedStableDiffusionPipeline.txt2img)
    def gen(self, prompt, **kwargs):
        self._to_device()
        with torch.autocast("cuda"):
            P = self.pipe.txt2img(prompt, **kwargs)
        P['index'] = len(self.saved)
        self.saved.append(P)
        if self.show_as_generated:
            display(self.with_prompt(P))
        return P
    
    @delegates(UnifiedStableDiffusionPipeline.txt2img)
    def generate_from_scratch(self, prompt, num=6, **kwargs):
        mb = master_bar(range(num))
        X = [self.gen(prompt, mb=mb, **kwargs) for _ in mb]
        
        return self._image_grid([self.with_prompt(x) for x in X])
    
    @delegates(UnifiedStableDiffusionPipeline.txt2img)
    def generate_variants(self, i, noise=0.3, num=8, prompt = None, steps=None, guidance_scale=None, vary_mean_std = False, **kwargs):
        I = self.saved[i]
        latents = I['latents']
        if prompt is None: prompt = self.prompt(i)
        if steps is None: steps = self.get_steps(i)
        if guidance_scale is None: guidance_scale = self.get_guidance_scale(i)
        
        lats = [self._randlat(latents=latents,noise=noise,vary_mean_std=vary_mean_std) for _ in range(num)]
        
        mb = master_bar(lats)
        X = [self.gen(prompt, latents=l, steps=steps, guidance_scale=guidance_scale, mb=mb, **kwargs) for l in mb]
        return self._image_grid([self.with_prompt(x) for x in [I]+X])
    
    @delegates(UnifiedStableDiffusionPipeline.txt2img)
    def generate_with_prompts(self, i, prompts, **kwargs):
        latents = self.saved[i]['latents']
        mb = master_bar(prompts)
        X = [self.gen(p, latents=latents, mb=mb, **kwargs) for p in mb]
        return self._image_grid([self.with_prompt(x) for x in X])
    
    @delegates(UnifiedStableDiffusionPipeline.txt2img)
    def interpolate(self, i, j, num=20, prompt=None, **kwargs):
        if prompt is None: prompt = self.saved[i]['prompt']
        Li, Lj = self.saved[i]['latents'], self.saved[j]['latents']
        mi, mj = Li.mean(), Lj.mean()
        si, sj = Li.std(), Lj.std()
        
        L = [torch.lerp(Li,Lj,p) for p in torch.linspace(0,1,num)]
        M = [torch.lerp(mi,mj,p) for p in torch.linspace(0,1,num)]
        S = [torch.lerp(si,sj,p) for p in torch.linspace(0,1,num)]
        
        L = [_with_mean_std(l,m,s) for l,m,s in zip(L,M,S)]
        
        mb = master_bar(L)
        X = [self.gen(prompt, latents=l, mb=mb, **kwargs) for l in mb]
        return self._image_grid([self.with_prompt(x) for x in X])
    
    @delegates(UnifiedStableDiffusionPipeline.txt2img)
    def incrust(self, i, num=1, start = (0,0),**kwargs):
        old_lat = self.saved[i]['latents']
        prompt = self.saved[i]['prompt']
        bs, c, oh, ow = old_lat.shape
        h, w = self.height//8, self.width//8
        if start == "center":
            sh, sw = (h-oh)//2, (w-ow)//2
        else:
            sh, sw = start
        X = []
        mb = master_bar(range(num))
        for _ in mb:
            lat = torch.randn(bs, c, h, w)
            lat[:,:,sh:sh+oh,sw:sw+ow] = old_lat[:,:,:oh,:ow]
            X += [self.gen(prompt,latents=lat,mb=mb,**kwargs)]
        return image_grid([self.with_prompt(x) for x in X])
    
    @delegates(UnifiedStableDiffusionPipeline.img2img)
    def img2img(self, img, prompt, mask = None, **kwargs):
        self._to_device()
        with torch.autocast("cuda"):
            P = self.pipe.img2img(prompt=prompt, img=img, mask=mask, **kwargs)
        P['index'] = len(self.saved)
        self.saved.append(P)
        return P
    
    @delegates(UnifiedStableDiffusionPipeline.img2img)
    def modify_image(self, img, prompt=None, num=6, **kwargs):
        if isinstance(img,int):
            if prompt is None: prompt = self.prompt(img)
            img = self.saved[img]['sample'][0]
        mb = master_bar(range(num))
        X = [self.img2img(prompt=prompt, img=img, mb=mb, **kwargs) for _ in mb]
        return self._image_grid([self.with_prompt(x) for x in X])
    
    #@delegates(UnifiedStableDiffusionPipeline.inpaint)
    def inpaint(self, img, prompt = None, mask = None, num=1, **kwargs):
        
        if isinstance(img,int): 
            if prompt is None: prompt = self.prompt(img)
            img = self.__getitem__(img)
        if isinstance(img,str) or isinstance(img,Path):
            img = Image.open(img)
        if isinstance(mask,str) or isinstance(mask,Path):
            mask = Image.open(mask)
        if mask is None:
            mask = mask_from_alpha(img)
            
        self._to_device(False)
        print(f"self.pipe.device = {self.pipe.device}")
        print(f"self.inpaint_pipe.device = {self.inpaint_pipe.device}")
        mb = master_bar(range(num))
        for _ in mb:
            with torch.autocast("cuda"):
                P = self.inpaint_pipe.inpaint(prompt, img=img, mask=mask, mb=mb,**kwargs)
            P['index'] = len(self.saved)
            self.saved.append(P)
        
        return self[-num:]
    
    #@delegates(UnifiedStableDiffusionPipeline.inpaint)
    def inpaint_gui(self, img = None, steps=None, num=1, strength=0.7, **kwargs):
        if steps is None: steps = self.pipe.steps
        prompt = ""
        if isinstance(img,int):
            prompt = self.saved[img]['prompt']
            img = self.saved[img]['sample'][0]
        if isinstance(img,str) or isinstance(img,Path):
            img = Image.open(img)
        with gr.Blocks() as block:
            col = gr.Column()
            
            def _inpaint_gui_out(self, prompt, negative_prompt, img_mask, num, steps, strength, pipe_name):
                if negative_prompt == "": negative_prompt = None
                col.update(visible=False)
                
                img = Image.fromarray(img_mask['image'])
                mask = Image.fromarray(img_mask['mask'])
                
                
                if pipe_name == "Inpaint":
                    out = self.inpaint(prompt=prompt, img=img, mask=mask, num=num, steps=steps, strength=strength, negative_prompt=negative_prompt, **kwargs)
                else:
                    out = self.modify_image(prompt=prompt, img=img, mask=mask, num=num, steps=steps, strength=strength, negative_prompt=negative_prompt, **kwargs)
                display(out)
                
                block.close()
                return out
            
            with col:
                txt = gr.Textbox(label="Prompt", value=prompt)
                ntxt = gr.Textbox(label="Negative Prompt")
                img_mask = gr.Image(value=img, tool='sketch')
                sld_num = gr.Slider(minimum=1,maximum=30,value=num,step=1,label="How many to generate")
                sld_steps = gr.Slider(minimum=10,maximum=150,value=steps,step=1,label="Num inference steps")
                sld_strength = gr.Slider(minimum=0,maximum=1,value=strength,step=0.05,label="Strength")
                rd_pipe = gr.Radio(["Img2img", "Inpaint"],value="Img2img",label="Which pipeline to use?")
                btn = gr.Button("Submit")
                btn.click(fn=lambda *args: _inpaint_gui_out(self, *args), inputs=[txt, ntxt, img_mask, sld_num, sld_steps, sld_strength, rd_pipe], outputs=None)
        block.launch(server_port=3123)
    
    def with_prompt(self, img, with_idx = True):
        if isinstance(img,int): img = self.saved[img]
        prompt = img['prompt']
        idx = img['index']
        steps = img['steps'] if 'steps' in img else 90
        guidance_scale = img['guidance_scale'] if 'guidance_scale' in img else 7.5
        
        img = img['sample'][0].copy()
        w,h = img.width, img.height
        drawer = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 18)
        
        p = textwrap.wrap(prompt,width=50)
        for i, t in enumerate(p):
            drawer.text((4, 4 + i*21), t, font=font, fill=(255, 0, 0, 200))
            
        drawer.text((w//2, h-40), str(idx), font=font, fill=(255, 255, 0))
        drawer.text((20, h-40), str(steps), font=font, fill=(0, 255, 0))
        drawer.text((w-50, h-40), f"{guidance_scale:.1f}", font=font, fill=(0, 0, 255))
        return img
    
    
    def get_images(self):
        return [x['sample'][0] for x in self.saved]
    
    def view_img(self,i,with_prompt=False):
        if with_prompt:
            return self.with_prompt(self.saved[i])
        return self.saved[i]['sample'][0]
    
    def __getitem__(self, i):
        if isinstance(i,int):
            print(self.prompt(i))
            return self.view_img(i)
        elif isinstance(i,list):
            return smart_image_grid([self.with_prompt(self.saved[j]) for j in i])
        else:
            return smart_image_grid([self.with_prompt(a) for a in self.saved[i]])
    
    def __len__(self):
        return len(self.saved)
    
    def save(self, i):
        if isinstance(i,list):
            for j in i:
                self.save(j)
        elif isinstance(i,int):
            I = self.saved[i]
            prompt = I['prompt'].replace("(","").replace(")","").replace("[","").replace("]","")[:128]
            fname = f"{prompt}_{i}_{random_string(4)}"
            I['filename'] = fname
            torch.save(I,self.savedir/f'pth/{fname}.pth')
            img = I['sample'][0]
            img.save(self.savedir/f"{fname}.png")
            print(f"Saved file {fname}")
        else:
            raise "Nope, only lists and ints"
    
    def load(self, fname):
        if type(fname) == str or type(fname) == Path:
            I = torch.load(fname)
            if 'filename' not in I:
                I['filename'] = Path(fname).stem
                print(f"filename not found in index, so re-saving it as {fname}.")
                torch.save(I,fname)
        else:
            I = fname
        I['index'] = len(self.saved)
        
        self.saved.append(I)
    
    def delete(self, i):
        if 'filename' not in self.saved[i]:
            print("Can't delete, isn't saved")
        else:
            fname = self.saved[i]['filename']
            pthfile = self.savedir/f'pth/{fname}.pth'
            pngfile = self.savedir/f'{fname}.png'
            print(f"Deleting {pthfile}")
            print(f"Deleting {pngfile}")
            pthfile.unlink()
            pngfile.unlink()
            
    
    def load_all(self, keywords=None):
        for f in os.scandir(self.savedir/"pth"):
            if f.name.split('.')[-1] == 'pth':
                if _are_in_prompt(keywords,f):
                    self.load(f.path)
                    
    def _image_grid(self,imgs):
        if self.show_as_generated:
            return
        return image_grid(imgs)
    
    def _randlat(self, latents, noise, vary_mean_std=False):
        N = torch.randn_like(latents)*noise
        m, s = latents.mean(), latents.std()
        if vary_mean_std:
            m *= 1+torch.randn_like(m)*noise
            s *= 1+torch.randn_like(m)*noise
        return _with_mean_std(latents+N, mean=m, std=s)
    
    def remove(self, i: int):
        self.saved[i] = self.saved[-1]
        self.saved[i]['index'] = i
        return self.saved.pop()
    
    def keep_if(self, f):
        n = len(self.saved)
        for i in range(n-1,-1,-1):
            if not f(self.saved[i]):
                self.remove(i)
    
    def redo_indices(self):
        for i, s in enumerate(self.saved):
            s['index'] = i
        
    def remove_if(self, f):
        return self.keep_if(lambda x: not f(x))
    
    def prompt(self, i):
        return self.saved[i]['prompt']
    
    def get_guidance_scale(self, i):
        if 'guidance_scale' in self.saved[i]:
            return self.saved[i]['guidance_scale']
        else:
            return 7.5
    
    def get_steps(self, i):
        if 'steps' in self.saved[i]:
            return self.saved[i]['steps']
        else:
            return 90
    
    @property
    def height(self):
        return self.pipe.height

    @height.setter
    def height(self, h: int):
        self.pipe.height = h
        
    @property
    def width(self):
        return self.pipe.width

    @width.setter
    def width(self, w: int):
        self.pipe.width = w
        
    @property
    def steps(self):
        return self.pipe.steps

    @steps.setter
    def steps(self, s: int):
        self.pipe.steps = s
        
    @property
    def guidance_scale(self):
        return self.pipe.guidance_scale

    @guidance_scale.setter
    def guidance_scale(self, g: int):
        self.pipe.guidance_scale = g
        
    def _to_device(self, original_pipe=True):
        if original_pipe:
            if self.pipe.unet.device == self.device: return
            self.inpaint_pipe.unet.to('cpu')
            self.inpaint_pipe.to('cpu')
            self.pipe.unet.to(self.device)
        else:
            if self.inpaint_pipe.unet.device == self.device: return
            self.pipe.unet.to('cpu')
            self.inpaint_pipe.unet.to(self.device)
            pritn(f"self.device = {self.device} but ")

def _are_in_prompt(keywords, f):
    if keywords is None: return True
    if type(keywords) == str: keywords = [keywords]
    prompt = torch.load(Path(f))['prompt']
    for k in keywords:
        if k.lower() not in prompt.lower():
            return False
    return True
