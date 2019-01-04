#======================================================================
# ====================================
# README file for Page Cropping component
# ====================================
# Filename : ocrd-anyBaseOCR-pagecropping.py

# Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
# Note:
# 1) this work has been done in DFKI, Kaiserslautern, Germany.
# 2) The parameters values are read from ocrd-anyBaseOCR-parameter.json file. The values can be changed in that file.
# 3) The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/). A sample image file (samples/becker_quaestio_1586_00013.tif) and mets.xml (work_dir/mets.xml) are provided. The sequence of operations is: binarization, deskewing, cropping and dewarping (or can also be: binarization, dewarping, deskewing, and cropping; depends upon use-case).

# *********** Method Behaviour ********************
# This function takes a document image as input and crops/selects the page content
# area only (that's mean remove textual noise as well as any other noise around page content area)
# *********** Method Behaviour ********************

# *********** LICENSE ********************
# Copyright 2018 Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Apache License 2.0

# A permissive license whose main conditions require preservation of copyright
# and license notices. Contributors provide an express grant of patent rights.
# Licensed works, modifications, and larger works may be distributed under
# different terms and without source code.

# *********** LICENSE ********************
#======================================================================
import sys,os,re,glob,argparse,codecs
import ocrolib
from ocrolib import hocr
import cv2
import numpy as np
import ntpath
from pylsd import lsd
import json
from xml.dom import minidom
from PIL import Image


def parseXML(fpath):
    input_files=[]
    xmldoc = minidom.parse(fpath)
    nodes = xmldoc.getElementsByTagName('mets:fileGrp')
    for attr in nodes:
        if attr.attributes['USE'].value==args.Input:
            childNodes = attr.getElementsByTagName('mets:FLocat')
            for f in childNodes:
                input_files.append(f.attributes['xlink:href'].value)
    return input_files

def write_to_xml(fpath):
    xmldoc = minidom.parse(args.mets)
    subRoot = xmldoc.createElement('mets:fileGrp')
    subRoot.setAttribute('USE', args.Output)

    for f in fpath:
        basefile = ocrolib.allsplitext(os.path.basename(f))[0]
        child = xmldoc.createElement('mets:file')
        child.setAttribute('ID', 'CROP_'+basefile)
        child.setAttribute('GROUPID', 'P_' + basefile)
        child.setAttribute('MIMETYPE', "image/png")

        subChild = xmldoc.createElement('mets:FLocat')
        subChild.setAttribute('LOCTYPE', "URL")
        subChild.setAttribute('xlink:href', f)

        subRoot.appendChild(child)
        child.appendChild(subChild)

    xmldoc.getElementsByTagName('mets:fileSec')[0].appendChild(subRoot);

    if not args.OutputMets:
        metsFileSave = open(os.path.join(args.work, os.path.basename(args.mets)), "w")
    else:
        metsFileSave = open(os.path.join(args.work, args.OutputMets if args.OutputMets.endswith(".xml") else args.OutputMets+'.xml'), "w")
    metsFileSave.write(xmldoc.toxml())

def write_crop_coordinate(base, coordinate):
    x1,y1,x2,y2 = coordinate
    with open(base + '-frame-pf.dat', 'w') as fp:
        fp.write(str(x1)+"\t"+str(y1)+"\t"+str(x2-x1)+"\t"+str(y2-y1))

def remove_rular(arg, base):
    basefile = ocrolib.allsplitext(os.path.basename(arg))[0]
    img = cv2.imread(arg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape
    imgArea = height*width

    ## Get bounding box x,y,w,h of each contours
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    rects = sorted(rects,key=lambda  x:(x[2]*x[3]),reverse=True)
    rects = [r for r in rects if (imgArea*args.maxRularArea)>(r[2]*r[3])>(imgArea*args.minRularArea)]    ## consider those rectangle whose area>10000 and less than one-fourth of images

    ## detect child rectangles. Usually those are not rular. Rular position are basically any one side.
    removeRect=[]
    for i, rect1 in enumerate(rects):
        (x1,y1,w1,h1)=rect1
        for rect2 in rects[i+1:len(rects)]:
            (x2,y2,w2,h2)=rect2
            if (x1<x2) and (y1<y2) and (x1+w1>x2+w2) and (y1+h1>y2+h2):
                removeRect.append(rect2)

    ## removed child rectangles.
    rects = [x for x in rects if x not in removeRect]

    predictRular=[]
    for rect in rects:
        (x,y,w,h)=rect
        if (w<width*args.rularWidth) and ((y>height*args.positionBelow) or ((x+w)<width*args.positionLeft) or (x>width*args.positionRight)):
            if (args.rularRatioMin<round(float(w)/float(h),2)<args.rularRatioMax) or (args.rularRatioMin<round(float(h)/float(w),2)<args.rularRatioMax):
                blackPixel = np.count_nonzero(img[y:y+h,x:x+w]==0)
                predictRular.append((x,y,w,h,blackPixel))

    ## Finally check number of black pixel to avoid false rular
    if predictRular:
        predictRular = sorted(predictRular,key=lambda  x:(x[4]),reverse=True)
        x,y,w,h,t = predictRular[0]
        cv2.rectangle(img, (x-15,y-15), (x+w+20,y+h+20), (255, 255, 255), cv2.FILLED)
    save_file_path = base + '.pf.png'
    cv2.imwrite(save_file_path, img)
    return save_file_path
# End of function remove_rular(arg)

def BorderLine(MaxBoundary, lines, index, flag):
    getLine=1
    LastLine=[]
    if flag=="top" or flag=="left":
        for i in range(len(lines)-1):
            if(abs(lines[i][index]-lines[i+1][index]))<=15 and lines[i][index]<MaxBoundary:
                LastLine=[lines[i][0], lines[i][1], lines[i][2], lines[i][3]]
                getLine+=1
            elif getLine>=3:
                break
            else:
                getLine=1
    elif flag=="bottom" or flag=="right":
        for i in reversed(list(range(len(lines)-1))):
            if(abs(lines[i][index]-lines[i+1][index]))<=15 and lines[i][index]>MaxBoundary:
                LastLine=[lines[i][0], lines[i][1], lines[i][2], lines[i][3]]
                getLine+=1
            elif getLine>=3:
                break
            else:
                getLine=1
    if getLine>=3 and LastLine:
        if flag=="top":
            lineDetectH.append((LastLine[0], max(LastLine[1],LastLine[3]), LastLine[2], max(LastLine[1],LastLine[3])))
        if flag=="left":
            lineDetectV.append((max(LastLine[0],LastLine[2]), LastLine[1], max(LastLine[0],LastLine[2]), LastLine[3]))
        if flag=="bottom":
            lineDetectH.append((LastLine[0], min(LastLine[1],LastLine[3]), LastLine[2], min(LastLine[1],LastLine[3])))
        if flag=="right":
            lineDetectV.append((min(LastLine[0],LastLine[2]), LastLine[1], min(LastLine[0], LastLine[2]), LastLine[3]))

def get_intersect(a1, a2, b1, b2):
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)
    if z == 0:
        #return (float('inf'), float('inf'))
        return (0, 0)
    return (x/z, y/z)

def detect_lines(arg):
    Hline=[]; Vline=[]
    img = cv2.imread(arg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgHeight, imgWidth = gray.shape
    lines = lsd(gray)

    for i in range(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))
        width = lines[i, 4]
        if (abs(pt1[0]-pt2[0])>45) and ((int(pt1[1])<imgHeight*0.25) or (int(pt1[1])>imgHeight*0.75)):        # consider those line whise length more than this orbitrary value
            Hline.append([0, int(pt1[1]), imgWidth, int(pt2[1])])    # make full horizontal line
        if (abs(pt1[1]-pt2[1])>45) and ((int(pt1[0])<imgWidth*0.4) or (int(pt1[0])>imgWidth*0.6)):
            Vline.append([int(pt1[0]), 0, int(pt2[0]), imgHeight])    # make full vertical line
    Hline.sort(key=lambda x:(x[1]), reverse=False)
    Vline.sort(key=lambda x:(x[0]), reverse=False)
    return img, imgHeight, imgWidth, Hline, Vline

def select_borderLine(arg, base):
    basefile = ocrolib.allsplitext(os.path.basename(arg))[0]
    img, imgHeight, imgWidth, Hlines, Vlines = detect_lines(arg)

    # top side
    BorderLine(imgHeight*0.25, Hlines, index=1, flag="top")
    # left side
    BorderLine(imgWidth*0.4, Vlines, index=0, flag="left")
    # bottom side
    BorderLine(imgHeight*0.75, Hlines, index=1, flag="bottom")
    # right side
    BorderLine(imgWidth*0.6, Vlines, index=0, flag="right")

    intersectPoint=[]
    for l1 in lineDetectH:
        for l2 in lineDetectV:
            x ,y = get_intersect((l1[0],l1[1]), (l1[2],l1[3]), (l2[0],l2[1]), (l2[2],l2[3]))
            intersectPoint.append([x,y])
    Xstart = 0; Xend = imgWidth; Ystart = 0; Yend = imgHeight
    for i in intersectPoint:
        Xs = int(i[0])+10 if i[0]<imgWidth*0.4 else 10
        if Xs>Xstart: Xstart = Xs
        Xe = int(i[0])-10 if i[0]>imgWidth*0.6 else int(imgWidth)-10
        if Xe<Xend: Xend = Xe
        Ys = int(i[1])+10 if i[1]<imgHeight*0.25 else 10
        #print("Ys,Ystart:",Ys,Ystart)
        if Ys>Ystart: Ystart = Ys
        Ye = int(i[1])-15 if i[1]>imgHeight*0.75 else int(imgHeight)-15
        if Ye<Yend: Yend = Ye

    if Xend<0: Xend = 10
    if Yend<0: Yend = 15
    save_pf(base, [Xstart,Ystart,Xend,Yend])

    return [Xstart,Ystart,Xend,Yend]

def filter_noisebox(textarea, height, width):
    tmp=[]; st=True

    while st==True:
        textarea = [list(x) for x in textarea if x not in tmp]
        tmp=[]
        textarea = sorted(textarea,key=lambda x:(x[3]),reverse=False)
        #print textarea
        x11,y11,x12,y12 = textarea[0]
        x21,y21,x22,y22 = textarea[1]

        if abs(y12-y21)>100 and (float(abs(x12-x11)*abs(y12-y11))/(height*width))<0.001:
            tmp.append(textarea[0])

        x11,y11,x12,y12 = textarea[-2]
        x21,y21,x22,y22 = textarea[-1]

        if abs(y12-y21)>100 and (float(abs(x21-x22)*abs(y22-y21))/(height*width))<0.001:
            tmp.append(textarea[-1])

        if len(tmp)==0:
            st=False

    return textarea

def detect_textarea(arg):
    textarea=[]
    noise_textarea=[]
    large = cv2.imread(arg)
    rgb = large
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    height, width = small.shape

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1)) # for historical docs
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    _, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        #print x,y,w,h
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        if r>0.45 and (width*0.9)>w>15 and (height*0.5)>h>15:
            textarea.append([x,y,x+w-1,y+h-1])
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 0, 255), 2)

    if len(textarea)>1:
        textarea = filter_noisebox(textarea, height, width)

    return textarea, rgb, height, width

def save_pf(base, textarea):
    x1,y1,x2,y2 = textarea

    img = Image.open(base+'.pf.png')
    img2 = img.crop((x1,y1,x2,y2))
    img2.save(base + '.pf.png')
    write_crop_coordinate(base, textarea)

def filter_area(textarea, binImg):
    height, width = binImg.shape
    tmp=[]
    for area in textarea:
        if (height*width*args.minArea <  (abs(area[2]-area[0]) * abs(area[3]-area[1]))):
            tmp.append(area)
    return tmp

def marge_columns(textarea, binImg):
    tmp=[]; marge=[]
    height, width = binImg.shape
    #print binImg.shape
    textarea.sort(key=lambda x:(x[0]))
    #print args.colSeparator
    for i in range(len(textarea)-1):
        st = False
        x11,y11,x12,y12 = textarea[i]
        x21,y21,x22,y22 = textarea[i+1]
        if x21-x12<=args.colSeparator:
            if len(marge)>0:
                #print "marge ", marge[0]
                x31,y31,x32,y32 = marge[0]
                marge.pop(0)
            else:
                x31,y31,x32,y32 = [9999,9999,0,0]
            marge.append([min(x11,x21,x31), min(y11,y21,y31), max(x12,x22,x32), max(y12,y22,y32)])
            st=True
        else:
            tmp.append(textarea[i])

    if st==False:
        tmp.append(textarea[-1])

    return tmp+marge

def crop_area(textarea, binImg, rgb, base):
    height, width = binImg.shape

    textarea = np.unique(textarea, axis=0)
    i=0
    tmp=[]; areas=[]
    while i<len(textarea):
        textarea = [list(x) for x in textarea if x not in tmp]
        tmp=[]
        if len(textarea)==0:
            break
        maxBox = textarea[0]
        for chkBox in textarea:
            if maxBox != chkBox:
                x11,y11,x12,y12 = maxBox
                x21,y21,x22,y22 = chkBox
                if ((x11<=x21<=x12) or (x21<=x11<=x22)):
                    tmp.append(maxBox)
                    tmp.append(chkBox)
                    maxBox = [min(x11,x21), min(y11,y21), max(x12,x22), max(y12,y22)]
        if len(tmp)==0: tmp.append(maxBox)
        x1,y1,x2,y2 = maxBox
        areas.append(maxBox)
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        i=i+1

    textarea = np.unique(areas, axis=0).tolist()
    if len(textarea)>0:
        textarea = filter_area(textarea, binImg)
    if len(textarea)>1:
        textarea = marge_columns(textarea, binImg)
        #print textarea

    if len(textarea)>0:
        textarea = sorted(textarea,key=lambda x:((x[2]-x[0])*(x[3]-x[1])),reverse=True)
        #print textarea
        x1,y1,x2,y2 = textarea[0]
        x1 = x1-20 if x1>20 else 0
        x2 = x2+20 if x2<width-20 else width
        y1 = y1-40 if y1>40 else 0
        y2 = y2+40 if y2<height-40 else height

        save_pf(base, [x1,y1,x2,y2])

    return textarea

def main():
    parser = argparse.ArgumentParser("""
    Image crop using non-linear processing.

        python ocrd-anyBaseOCR-cropping.py -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)

    """)

    parser.add_argument('-p','--parameter',type=str,help="Parameter file location")
    parser.add_argument('--maxRularArea',type=float,help='Consider maximum rular area')
    parser.add_argument('--minRularArea',type=float,help='Consider minimum rular area')
    parser.add_argument('--rularWidth',type=float,help='maximum rular width')
    parser.add_argument('--positionLeft',type=float,help='rular position in left')
    parser.add_argument('--positionRight',type=float,help='rular position in right')
    parser.add_argument('--positionBelow',type=float,help='rular position in below')
    parser.add_argument('--rularRatioMax',type=float,help='rular position in below')
    parser.add_argument('--rularRatioMin',type=float,help='rular position in below')
    parser.add_argument('--colSeparator',type=float, help='consider space between column. 25% of width')
    parser.add_argument('--minArea',type=float, help='rular position in below')
    #parser.add_argument('files',nargs='+')
    parser.add_argument('-O','--Output',default=None,help="output directory")
    parser.add_argument('-w','--work',type=str,help="Working directory location", default=".")
    parser.add_argument('-I','--Input',default=None,help="Input directory")
    parser.add_argument('-m','--mets',default=None,help="METs input file")
    parser.add_argument('-o','--OutputMets',default=None,help="METs output file")
    parser.add_argument('-g','--group',default=None,help="METs image group id")

    args = parser.parse_args()

    ## Read parameter values from json file
    if args.parameter:
        if not os.path.exists(args.parameter):
            print("Error : Parameter file does not exists.")
            sys.exit(0)
        else:
            param = json.load(open(args.parameter))
    else:
        if not os.path.exists('ocrd-anyBaseOCR-parameter.json'):
            print("Error : Parameter file does not exists.")
            sys.exit(0)
        else:
            param = json.load(open('ocrd-anyBaseOCR-parameter.json'))

    args.maxRularArea = param["anyBaseOCR"]["crop"]["maxRularArea"]
    args.minRularArea = param["anyBaseOCR"]["crop"]["minRularArea"]
    args.rularWidth = param["anyBaseOCR"]["crop"]["rularWidth"]
    args.positionLeft = param["anyBaseOCR"]["crop"]["positionLeft"]
    args.positionRight = param["anyBaseOCR"]["crop"]["positionRight"]
    args.positionBelow = param["anyBaseOCR"]["crop"]["positionBelow"]
    args.rularRatioMax = param["anyBaseOCR"]["crop"]["rularRatioMax"]
    args.rularRatioMin = param["anyBaseOCR"]["crop"]["rularRatioMin"]
    args.colSeparator = param["anyBaseOCR"]["crop"]["colSeparator"]
    args.minArea = param["anyBaseOCR"]["crop"]["minArea"]

    ### End to read parameters
    # mendatory parameter check
    if not args.mets or not args.Input or not args.Output or not args.work:
        parser.print_help()
        print("Example: python ocrd-anyBaseOCR-cropping.py -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)")
        sys.exit(0)

    if args.work:
        if not os.path.exists(args.work):
            os.mkdir(args.work)

    files = parseXML(args.mets)
    fname=[]
    for i, f in enumerate(files):
        print("Process file: ", str(f) , i+1)
        base,_ = ocrolib.allsplitext(str(f))
        binImg = ocrolib.read_image_binary(str(f))

        lineDetectH=[]; lineDetectV=[]
        fpath = remove_rular(str(f), base)
        textarea, rgb, height, width = detect_textarea(fpath)
        args.colSeparator = int(width * args.colSeparator)

        if len(textarea)>1:
            textarea = crop_area(textarea, binImg, rgb, base)
            if len(textarea)==0:
                select_borderLine(fpath, base)
        elif len(textarea)==1 and (height*width*0.5 <  (abs(textarea[0][2]-textarea[0][0]) * abs(textarea[0][3]-textarea[0][1]))):
            x1,y1,x2,y2 = textarea[0]
            x1 = x1-20 if x1>20 else 0
            x2 = x2+20 if x2<width-20 else width
            y1 = y1-40 if y1>40 else 0
            y2 = y2+40 if y2<height-40 else height

            save_pf(base, [x1,y1,x2,y2])
        else:
            select_borderLine(fpath, base)

        fname.append(base + '.pf.png')
    write_to_xml(fname)
