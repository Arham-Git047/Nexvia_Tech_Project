from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / 'diagrams' / 'system_architecture.png'
OUT.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(15, 7.7), dpi=180)
fig.patch.set_facecolor('#f4f7fa'); ax.set_facecolor('#f4f7fa')
ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis('off')

def box(x,y,w,h,title,body,fill='#ffffff',edge='#c8d7e0', title_color='#173a52'):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=0.12',linewidth=1.35,edgecolor=edge,facecolor=fill)
    ax.add_patch(p)
    ax.text(x+w/2,y+h-0.33,title,ha='center',va='center',fontsize=11.4,fontweight='bold',color=title_color)
    ax.text(x+w/2,y+h-0.76,body,ha='center',va='top',fontsize=8.4,color='#587184',linespacing=1.35,wrap=True)

def arrow(p1,p2,label='',color='#4e8796'):
    a=FancyArrowPatch(p1,p2,arrowstyle='-|>',mutation_scale=14,linewidth=1.6,color=color,connectionstyle='arc3,rad=0.0')
    ax.add_patch(a)
    if label:
        mx=(p1[0]+p2[0])/2; my=(p1[1]+p2[1])/2+0.18
        ax.text(mx,my,label,ha='center',va='center',fontsize=8,color='#386f7e',bbox=dict(boxstyle='round,pad=0.16',facecolor='#f4f7fa',edgecolor='none'))

ax.text(.25,7.52,'MEDVISION REVIEW',fontsize=17,fontweight='bold',color='#12344b')
ax.text(.25,7.14,'Research prototype architecture · inference fail-closed until trained weights are supplied',fontsize=10,color='#658090')
# central row
box(.3,4.55,2.25,1.55,'Upload / DICOM','React reviewer console\nPNG / JPEG export\nOrthanc REST preview',fill='#eaf4f7',edge='#8bbcc8')
box(3.25,4.55,2.35,1.55,'FastAPI Gateway','MIME / size checks\nModel status\nCase & review endpoints',fill='#ffffff')
box(6.35,4.55,2.5,1.55,'Inference Engine','DenseNet121 · 14 labels\nGrad-CAM activation map\n503 if checkpoint missing',fill='#e8f6f1',edge='#82b9a2')
box(9.6,4.55,2.15,1.55,'Human Review','Approve / reject\nReviewer note\nCase state transition',fill='#fff4da',edge='#dfbf6a')
box(12.45,4.55,2.15,1.55,'Audit & Storage','JSON case registry\nImage / heatmap files\nAppend-only JSONL events',fill='#f0edf9',edge='#b3a4d8')
arrow((2.55,5.32),(3.25,5.32),'HTTPS / local')
arrow((5.6,5.32),(6.35,5.32),'validated')
arrow((8.85,5.32),(9.6,5.32),'findings')
arrow((11.75,5.32),(12.45,5.32),'decision')
# auxiliary row
box(1.0,1.25,3.0,1.65,'Public research data','NIH ChestX-ray14\npatient-level splits\nDVC metadata / hashes',fill='#ffffff')
box(5.25,1.25,3.2,1.65,'Training pipelines','PyTorch DenseNet121\nMONAI 3D U-Net (MSD)\nMetrics only after evaluation',fill='#ffffff')
box(9.8,1.25,3.2,1.65,'Deployment boundary','Docker Compose local demo\nOrthanc optional profile\nNo authentication / no clinical use',fill='#ffffff')
arrow((4.0,2.05),(5.25,2.05),'train')
arrow((8.45,2.05),(9.8,2.05),'checkpoint')
arrow((6.85,2.9),(7.55,4.55),'trained weights',color='#79a58e')
arrow((11.4,2.9),(10.75,4.55),'local runtime',color='#8d88bb')
ax.text(.3,.42,'Security note: Ports are loopback-bound in Compose. Use de-identified research data only; add auth, TLS, access control and validated preprocessing before any shared deployment.',fontsize=8.7,color='#758995')
plt.tight_layout(pad=1.2)
fig.savefig(OUT,bbox_inches='tight')
plt.close(fig)
print(OUT)
