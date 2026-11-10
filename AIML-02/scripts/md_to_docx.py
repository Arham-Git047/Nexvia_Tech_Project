from pathlib import Path
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT.parent
DOCS = [
    ('docs/system_architecture_api.md','System_Architecture_API_Documentation.docx'),
    ('docs/test_results.md','Test_Results_and_Monitoring_Report.docx'),
    ('docs/model_card.md','Model_Card_and_Limitations.docx'),
    ('docs/individual_reflection_draft.md','Individual_Reflection_Draft.docx'),
]
NAVY='102A43'; TEAL='2C8290'; LIGHT='EAF3F6'; LINE='D9E4EA'; MUTED='657D8D'; WHITE='FFFFFF'

def set_cell_shading(cell, fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'), fill); tcPr.append(shd)

def set_cell_border(cell, **kwargs):
    tc = cell._tc; tcPr = tc.get_or_add_tcPr(); tcBorders = tcPr.first_child_found_in('w:tcBorders')
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders'); tcPr.append(tcBorders)
    for edge in ('top','left','bottom','right','insideH','insideV'):
        if edge in kwargs:
            tag='w:{}'.format(edge); element=tcBorders.find(qn(tag))
            if element is None: element=OxmlElement(tag); tcBorders.append(element)
            for key,val in kwargs[edge].items(): element.set(qn('w:'+key), str(val))

def set_run(run, size=10, color=NAVY, bold=False, italic=False, font='Aptos'):
    run.font.name=font; run._element.rPr.rFonts.set(qn('w:eastAsia'),font)
    run.font.size=Pt(size); run.font.color.rgb=RGBColor.from_string(color)
    run.bold=bold; run.italic=italic

def add_page_field(paragraph):
    r=paragraph.add_run(); set_run(r, 9, MUTED)
    fldChar1=OxmlElement('w:fldChar'); fldChar1.set(qn('w:fldCharType'),'begin')
    instrText=OxmlElement('w:instrText'); instrText.set(qn('xml:space'),'preserve'); instrText.text=' PAGE '
    fldChar2=OxmlElement('w:fldChar'); fldChar2.set(qn('w:fldCharType'),'end')
    r._r.append(fldChar1); r._r.append(instrText); r._r.append(fldChar2)

def add_inline(paragraph, text):
    # Simple inline markdown support: bold, inline code, emphasis. URLs remain visible/clickable text in Word.
    pattern=re.compile(r'(\*\*.*?\*\*|`[^`]+`|\*[^*]+\*)')
    pos=0
    for m in pattern.finditer(text):
        if m.start()>pos:
            r=paragraph.add_run(text[pos:m.start()]); set_run(r,9.5,NAVY)
        token=m.group()
        if token.startswith('**'):
            r=paragraph.add_run(token[2:-2]); set_run(r,9.5,NAVY,True)
        elif token.startswith('`'):
            r=paragraph.add_run(token[1:-1]); set_run(r,9,TEAL,False,False,'Consolas')
        else:
            r=paragraph.add_run(token[1:-1]); set_run(r,9.5,NAVY,False,True)
        pos=m.end()
    if pos<len(text):
        r=paragraph.add_run(text[pos:]); set_run(r,9.5,NAVY)

def is_table_row(line): return line.strip().startswith('|') and line.strip().endswith('|')

def parse_table_cells(line): return [x.strip() for x in line.strip().strip('|').split('|')]

def create_doc(md_path, out_path):
    lines=md_path.read_text(encoding='utf-8').splitlines()
    doc=Document(); sec=doc.sections[0]
    sec.top_margin=Inches(.58); sec.bottom_margin=Inches(.58); sec.left_margin=Inches(.72); sec.right_margin=Inches(.72)
    sec.header_distance=Inches(.32); sec.footer_distance=Inches(.32)
    styles=doc.styles
    styles['Normal'].font.name='Aptos'; styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'),'Aptos'); styles['Normal'].font.size=Pt(9.5); styles['Normal'].font.color.rgb=RGBColor.from_string(NAVY)
    styles['Normal'].paragraph_format.space_after=Pt(3); styles['Normal'].paragraph_format.line_spacing=1.0
    for nm,size,col in [('Heading 1',17,NAVY),('Heading 2',12,TEAL),('Heading 3',10,NAVY)]:
        st=styles[nm]; st.font.name='Aptos Display'; st._element.rPr.rFonts.set(qn('w:eastAsia'),'Aptos Display'); st.font.size=Pt(size); st.font.bold=True; st.font.color.rgb=RGBColor.from_string(col); st.paragraph_format.space_before=Pt(8); st.paragraph_format.space_after=Pt(3); st.paragraph_format.keep_with_next=True
    header=sec.header.paragraphs[0]; header.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    rr=header.add_run('MEDVISION REVIEW  |  RESEARCH PROTOTYPE'); set_run(rr,8,TEAL,True)
    footer=sec.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    rr=footer.add_run('Arham Kelkar  •  Dr. D.Y. Patil School of Science and Technology  •  Page '); set_run(rr,8,MUTED)
    add_page_field(footer)
    i=0; first_heading=True
    while i<len(lines):
        line=lines[i].rstrip()
        if not line.strip(): i+=1; continue
        if line.startswith('```'):
            code=[]; i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]); i+=1
            p=doc.add_paragraph(); p.paragraph_format.left_indent=Inches(.12); p.paragraph_format.right_indent=Inches(.12); p.paragraph_format.space_before=Pt(3); p.paragraph_format.space_after=Pt(7)
            pPr=p._p.get_or_add_pPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),'F1F5F7'); pPr.append(shd)
            r=p.add_run('\n'.join(code)); set_run(r,8,'2C5268',False,False,'Consolas')
            i+=1; continue
        if is_table_row(line):
            table_lines=[]
            while i<len(lines) and is_table_row(lines[i]):
                table_lines.append(parse_table_cells(lines[i])); i+=1
            if len(table_lines)>1 and all(re.fullmatch(r':?-{3,}:?',cell.replace(' ','')) for cell in table_lines[1]):
                table_lines.pop(1)
            ncols=max(len(x) for x in table_lines); table=doc.add_table(rows=len(table_lines), cols=ncols); table.alignment=WD_TABLE_ALIGNMENT.CENTER; table.autofit=True
            for ri,row in enumerate(table_lines):
                for ci in range(ncols):
                    cell=table.cell(ri,ci); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    cell.text=''
                    p=cell.paragraphs[0]; p.paragraph_format.space_after=Pt(1); p.paragraph_format.space_before=Pt(1)
                    txt=row[ci] if ci<len(row) else ''; add_inline(p,txt)
                    for r in p.runs:
                        if ri==0: r.font.bold=True; r.font.color.rgb=RGBColor.from_string(WHITE)
                        r.font.size=Pt(7.8)
                    set_cell_shading(cell, NAVY if ri==0 else ('F5F8FA' if ri%2 else WHITE))
                    set_cell_border(cell,top={'val':'single','sz':'3','color':LINE},bottom={'val':'single','sz':'3','color':LINE},left={'val':'single','sz':'3','color':LINE},right={'val':'single','sz':'3','color':LINE})
            doc.add_paragraph().paragraph_format.space_after=Pt(1)
            continue
        if line.startswith('# '):
            title=line[2:].strip()
            p=doc.add_paragraph(style='Title'); p.paragraph_format.space_after=Pt(7); p.paragraph_format.keep_with_next=True
            r=p.add_run(title); set_run(r,22,NAVY,True,False,'Aptos Display')
            first_heading=False; i+=1; continue
        if line.startswith('## '):
            p=doc.add_paragraph(style='Heading 1'); add_inline(p,line[3:].strip()); i+=1; continue
        if line.startswith('### '):
            p=doc.add_paragraph(style='Heading 2'); add_inline(p,line[4:].strip()); i+=1; continue
        if line.startswith('#### '):
            p=doc.add_paragraph(style='Heading 3'); add_inline(p,line[5:].strip()); i+=1; continue
        if line.startswith('> '):
            p=doc.add_paragraph(); p.paragraph_format.left_indent=Inches(.18); p.paragraph_format.right_indent=Inches(.08); p.paragraph_format.space_before=Pt(3); p.paragraph_format.space_after=Pt(8)
            pPr=p._p.get_or_add_pPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),'FFF4D8'); pPr.append(shd)
            add_inline(p,line[2:].strip())
            for r in p.runs: r.font.color.rgb=RGBColor.from_string('77590A'); r.font.bold=True
            i+=1; continue
        if line.startswith('- ') or line.startswith('* '):
            p=doc.add_paragraph(style='List Bullet'); p.paragraph_format.space_after=Pt(1); add_inline(p,line[2:].strip()); i+=1; continue
        m=re.match(r'^(\d+)\.\s+(.*)$',line)
        if m:
            p=doc.add_paragraph(style='List Number'); p.paragraph_format.space_after=Pt(1); add_inline(p,m.group(2)); i+=1; continue
        # regular paragraphs; join following wrapped non-special lines
        para=[line]; i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','> ','- ','* ','```')) and not is_table_row(lines[i]) and not re.match(r'^\d+\.\s+',lines[i]):
            para.append(lines[i].strip()); i+=1
        p=doc.add_paragraph(); add_inline(p,' '.join(para))
    doc.core_properties.author='Arham Kelkar'
    doc.core_properties.title=md_path.stem.replace('_',' ').title()
    doc.core_properties.subject='AI-Assisted Medical Imaging and Diagnosis System — research prototype'
    doc.save(out_path)
    print(out_path)

for relative,name in DOCS:
    create_doc(ROOT/relative, OUTDIR/name)
