import torch
from math import floor, ceil, sqrt
from PIL import Image, ImageDraw, ImageFont
import textwrap
from fastprogress import progress_bar, master_bar
from unified_pipeline import UnifiedStableDiffusionPipeline
import random
import os
import pathlib
import numpy as np
from pathlib import Path
import gradio as gr
from IPython.display import display
from delegation import delegates
from math import prod

def image_grid(imgs):
    num = len(imgs)
    cols = min(4,ceil(sqrt(num)))
    rows = ceil(num/cols)
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def random_string(n):
    X = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join([random.choice(X) for _ in range(n)])

def get_image_and_mask(img):
    a = np.array(img).astype(np.uint8)
    assert(a.shape[2] == 4)
    img, mask = a[:,:,:3], a[:,:,3:]
    mask = mask.repeat(3,axis=2)
    return img, mask

class EnhancedGenerator:
    def __init__(self, pipe, height=512, width=768, savedir = "saved"):
        self.pipe = pipe
        self.height = height
        self.width = width
        self.saved = []
        
        self.savedir = Path(savedir)
        (self.savedir/"pth").mkdir(exist_ok=True, parents=True)
        self.show_as_generated = False   
        
    @delegates(UnifiedStableDiffusionPipeline.generate)
    def gen(self, prompt, **kwargs):
        with torch.autocast("cuda"):
            P = self.pipe.generate(prompt, **kwargs)
        P['prompt'] = prompt
        P['index'] = len(self.saved)
        self.saved.append(P)
        return P
    
    @delegates(UnifiedStableDiffusionPipeline.generate)
    def generate_from_scratch(self, prompt, num=6, **kwargs):
        mb = master_bar(range(num))
        X = [self.gen(prompt, mb=mb, **kwargs) for _ in mb]
        
        return self._image_grid([self.with_prompt(x) for x in X])
    
    @delegates(UnifiedStableDiffusionPipeline.generate)
    def generate_variants(self, i, noise=0.4, num=6, prompt = None, **kwargs):
        I = self.saved[i]
        latents = I['latents']
        if prompt is None: prompt = I['prompt']
        
        lats = [self._randlat(latents,noise) for _ in range(num)]
        
        mb = master_bar(lats)
        X = [self.gen(prompt, latents=l, mb=mb,**kwargs) for l in mb]
        return self._image_grid([self.with_prompt(x) for x in [I]+X])
    
    @delegates(UnifiedStableDiffusionPipeline.generate)
    def generate_with_prompts(self, i, prompts, **kwargs):
        latents = self.saved[i]['latents']
        mb = master_bar(prompts)
        X = [self.gen(p, latents=latents, mb=mb, **kwargs) for p in mb]
        return self._image_grid([self.with_prompt(x) for x in X])
    
    @delegates(UnifiedStableDiffusionPipeline.generate)
    def interpolate(self, i, j, num=20, prompt=None, **kwargs):
        if prompt is None: prompt = self.saved[i]['prompt']
        Li, Lj = self.saved[i]['latents'], self.saved[j]['latents']
        L = [torch.lerp(Li,Lj,p.item()) for p in torch.linspace(0,1,num)]
        L = [(l-l.mean())/l.std() for l in L]
        mb = master_bar(L)
        X = [self.gen(prompt, latents=l, mb=mb, **kwargs) for l in mb]
        return self._image_grid([self.with_prompt(x) for x in X])
    
    @delegates(UnifiedStableDiffusionPipeline.generate)
    def modify_image(self, img, prompt=None, num=6, iterations=1, **kwargs):
        if isinstance(img,str) or isinstance(img,Path): img = Image.open(img)
        if isinstance(img,int):
            if prompt is None:
                prompt = self.saved[img]['prompt']
            img = self.saved[img]['sample'][0]
        X = [self.gen(prompt, img=img) for _ in progress_bar(range(num))]
        for i in range(1,iterations):
            print(f"Doing step {i+1}/{iterations}")
            X = [self.gen(prompt, img=x['sample'][0], **kwargs) for x in progress_bar(X)]
        return self._image_grid([self.with_prompt(x) for x in X])
    
    @delegates(UnifiedStableDiffusionPipeline.inpaint)
    def inpaint(self, prompt, img, mask, num=1, **kwargs):
        if isinstance(img,np.ndarray):
            img = Image.fromarray(img)
        if isinstance(mask,np.ndarray):
            mask = Image.fromarray(mask)
        
        mb = master_bar(range(num))
        for _ in mb:
            with torch.autocast("cuda"):
                P = self.pipe.inpaint(prompt, img=img, mask=mask, mb=mb, **kwargs)
            P['prompt'] = prompt
            P['index'] = len(self.saved)
            self.saved.append(P)
        
        return self[-num:]
    
    @delegates(UnifiedStableDiffusionPipeline.inpaint)
    def inpaint_gui(self, img = None, steps=80, num=1, **kwargs):
        prompt = ""
        if isinstance(img,int):
            prompt = self.saved[img]['prompt']
            img = self.saved[img]['sample'][0]
        if isinstance(img,str) or isinstance(img,Path):
            img = Image.open(img)
        with gr.Blocks() as block:
            
            def _inpaint_gui_out(self, imgmask, num, steps, prompt):
                out = self.inpaint(prompt, img=imgmask['image'], mask=imgmask['mask'], num=num, steps=steps, **kwargs)
                display(out)
                block.clear()
                block.close()
                return out
            
            with gr.Column():
                txt = gr.Textbox(label="Prompt", value=prompt)
                inp = gr.Image(value=img, tool='sketch')
                sld_num = gr.Slider(minimum=1,maximum=20,value=num,step=1,label="How many to generate")
                sld_steps = gr.Slider(minimum=10,maximum=150,value=steps,step=1,label="Num inference steps")

                btn = gr.Button("Submit")
                btn.click(fn=lambda *args: _inpaint_gui_out(self, *args), inputs=[inp, sld_num, sld_steps, txt], outputs=None)
        block.launch(server_port=3123)
    
    
    
    def with_prompt(self, img, with_idx = True):
        prompt = img['prompt']
        idx = img['index']
        img = img['sample'][0].copy()
        w,h = img.width, img.height
        drawer = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 18)
        
        p = textwrap.wrap(prompt,width=50)
        for i, t in enumerate(p):
            drawer.text((4, 4 + i*21), t, font=font, fill=(255, 0, 0, 200))
            
        drawer.text((w//2, h-40), str(idx), font=font, fill=(255, 255, 0))
        return img
    
    def get_images(self):
        return [x['sample'][0] for x in self.saved]
    
    def view_img(self,i,with_prompt=False):
        if with_prompt:
            return self.with_prompt(self.saved[i])
        return self.saved[i]['sample'][0]
    
    def __getitem__(self, i):
        if isinstance(i,int):
            return self.view_img(i)
        else:
            return image_grid([self.with_prompt(a) for a in self.saved[i]])
    
    def __len__(self):
        return len(self.saved)
    
    def save(self, i):
        if isinstance(i,list):
            for j in i:
                self.save(j)
        elif isinstance(i,int):
            I = self.saved[i]
            prompt = I['prompt']
            fname = f"{prompt}_{i}_{random_string(4)}"
            torch.save(I,self.savedir/f'pth/{fname}.pth')
            img = I['sample'][0]
            img.save(self.savedir/f"{fname}.png")
        else:
            raise "Nope, only lists and ints"
    
    def load(self, I):
        if type(I) == str or type(I) == Path:
            I = torch.load(I)
        I['index'] = len(self.saved)
        self.saved.append(I)
    
    def load_all(self, keyword=None):
        for f in os.scandir(self.savedir/"pth"):
            if f.name.split('.')[-1] == 'pth':
                if keyword is None or keyword.lower() in f.path.lower():
                    self.load(f.path)
                    
    def _image_grid(self,imgs):
        if self.show_as_generated:
            return
        return image_grid(imgs)
    
    def _randlat(self, latents, noise):
        factor = sqrt(prod(latents.shape)-1)
        mu = torch.randn((1,),device=latents.device)/factor
        sigma = torch.randn((1,),device=latents.device)/factor
        
        N = torch.randn_like(latents)*noise
        return (N-N.mean())/(N.std()+sigma) + mu
    
    @property
    def height(self):
        return self.pipe.height

    @height.setter
    def height(self, h: int):
        self.pipe.height = h
    
    @height.deleter
    def height(self):
        del self.pipe.height
        
    @property
    def width(self):
        return self.pipe.width

    @width.setter
    def width(self, w: int):
        self.pipe.width = w
