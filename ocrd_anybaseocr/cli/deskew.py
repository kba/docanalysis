#======================================================================
# ====================================
# README file for Skew Correction component
# ====================================

# Filename : ocrd-anyBaseOCR-deskew.py

# Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
# Note:
# 1) this work has been done in DFKI, Kaiserslautern, Germany.
# 2) The parameters values are read from ocrd-anyBaseOCR-parameter.json file. The values can be changed in that file.
# 3) The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/). A sample image file (samples/becker_quaestio_1586_00013.tif) and mets.xml (work_dir/mets.xml) are provided. The sequence of operations is: binarization, deskewing, cropping and dewarping (or can also be: binarization, dewarping, deskewing, and cropping; depends upon use-case).

# *********** Method Behaviour ********************
# This function takes a document image as input and do the skew correction of that document.
# *********** Method Behaviour ********************

# *********** LICENSE ********************
# License: ocropus-nlbin.py (from https://github.com/tmbdev/ocropy/) contains both functionalities: binarization and skew correction.
# This method (ocrd-anyBaseOCR-deskew.py) only contains the skew correction functionality of ocropus-nlbin.py.
# It still has the same licenses as ocropus-nlbin, i.e Apache 2.0 (the ocropy license details are pasted below).
# This file is dependend on ocrolib library which comes from https://github.com/tmbdev/ocropy/.

#Copyright 2014 Thomas M. Breuel

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
# limitations under the License.

#*********** LICENSE ********************
#======================================================================
#!/usr/bin/env python


from pylab import *
from numpy.ctypeslib import ndpointer
import argparse,os,os.path,glob
from scipy.ndimage import filters,interpolation,morphology,measurements
from scipy import stats
import multiprocessing
import ocrolib
import json
from xml.dom import minidom

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
        #basefile = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
        basefile = ocrolib.allsplitext(os.path.basename(f))[0]
        child = xmldoc.createElement('mets:file')
        child.setAttribute('ID', 'DESKEW_'+basefile)
        child.setAttribute('GROUPID', 'P_' + basefile)
        child.setAttribute('MIMETYPE', "image/png")

        subChild = xmldoc.createElement('mets:FLocat')
        subChild.setAttribute('LOCTYPE', "URL")
        subChild.setAttribute('xlink:href', f)

        #xmldoc.getElementsByTagName('mets:file')[0].appendChild(subChild);
        subRoot.appendChild(child)
        child.appendChild(subChild)

    #subRoot.appendChild(child)
    xmldoc.getElementsByTagName('mets:fileSec')[0].appendChild(subRoot);

    if not args.OutputMets:
        metsFileSave = open(os.path.join(args.work, os.path.basename(args.mets)), "w")
    else:
        metsFileSave = open(os.path.join(args.work, args.OutputMets if args.OutputMets.endswith(".xml") else args.OutputMets+'.xml'), "w")
    metsFileSave.write(xmldoc.toxml())

#args.files = ocrolib.glob_all(args.files)

def print_info(*objs):
    print("INFO: ", *objs, file=sys.stdout)

def estimate_skew_angle(image,angles):
    estimates = []
    for a in angles:
        v = mean(interpolation.rotate(image,a,order=0,mode='constant'),axis=1)
        v = var(v)
        estimates.append((v,a))
    if args.debug>0:
        plot([y for x,y in estimates],[x for x,y in estimates])
        ginput(1,args.debug)
    _,a = max(estimates)
    return a

def deskew(fpath, job):
    base,_ = ocrolib.allsplitext(fpath)
    basefile = ocrolib.allsplitext(os.path.basename(fpath))[0]

    if args.parallel<2: print_info("=== %s %-3d" % (fpath, job))
    raw = ocrolib.read_image_gray(fpath)

    flat = raw
    # estimate skew angle and rotate
    if args.maxskew>0:
        if args.parallel<2: print_info("estimating skew angle")
        d0,d1 = flat.shape
        o0,o1 = int(args.bignore*d0),int(args.bignore*d1)
        flat = amax(flat)-flat
        flat -= amin(flat)
        est = flat[o0:d0-o0,o1:d1-o1]
        ma = args.maxskew
        ms = int(2*args.maxskew*args.skewsteps)
        angle = estimate_skew_angle(est,linspace(-ma,ma,ms+1))
        flat = interpolation.rotate(flat,angle,mode='constant',reshape=0)
        flat = amax(flat)-flat
    else:
        angle = 0

    # estimate low and high thresholds
    if args.parallel<2: print_info("estimating thresholds")
    d0,d1 = flat.shape
    o0,o1 = int(args.bignore*d0),int(args.bignore*d1)
    est = flat[o0:d0-o0,o1:d1-o1]
    if args.escale>0:
        # by default, we use only regions that contain
        # significant variance; this makes the percentile
        # based low and high estimates more reliable
        e = args.escale
        v = est-filters.gaussian_filter(est,e*20.0)
        v = filters.gaussian_filter(v**2,e*20.0)**0.5
        v = (v>0.3*amax(v))
        v = morphology.binary_dilation(v,structure=ones((int(e*50),1)))
        v = morphology.binary_dilation(v,structure=ones((1,int(e*50))))
        if args.debug>0: imshow(v); ginput(1,args.debug)
        est = est[v]
    lo = stats.scoreatpercentile(est.ravel(),args.lo)
    hi = stats.scoreatpercentile(est.ravel(),args.hi)
    # rescale the image to get the gray scale image
    if args.parallel<2: print_info("rescaling")
    flat -= lo
    flat /= (hi-lo)
    flat = clip(flat,0,1)
    if args.debug>0: imshow(flat,vmin=0,vmax=1); ginput(1,args.debug)
    bin = 1*(flat>args.threshold)

    # output the normalized grayscale and the thresholded images
    print_info("%s lo-hi (%.2f %.2f) angle %4.1f" % (basefile, lo, hi, angle))
    if args.parallel<2: print_info("writing")
    ocrolib.write_image_binary(base+".ds.png",bin)
    return base+".ds.png"

def main():
    parser = argparse.ArgumentParser("""
    Image deskewing using non-linear processing.

        python ocrd-anyBaseOCR-deskew.py -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)

    This is a compute-intensive deskew method that works on degraded and historical book pages.
    """)

    parser.add_argument('-p','--parameter',type=str,help="Parameter file location")
    parser.add_argument('-e','--escale',type=float,help='scale for estimating a mask over the text region, default: %(default)s')
    parser.add_argument('-t','--threshold',type=float,help='threshold, determines lightness, default: %(default)s')
    parser.add_argument('-b','--bignore',type=float,help='ignore this much of the border for threshold estimation, default: %(default)s')
    parser.add_argument('-ms','--maxskew',type=float,help='skew angle estimation parameters (degrees), default: %(default)s')
    parser.add_argument('--skewsteps',type=int,help='steps for skew angle estimation (per degree), default: %(default)s')
    parser.add_argument('--debug',type=float,help='display intermediate results, default: %(default)s')
    parser.add_argument('--lo',type=float,help='percentile for black estimation, default: %(default)s')
    parser.add_argument('--hi',type=float,help='percentile for white estimation, default: %(default)s')
    parser.add_argument('-Q','--parallel',type=int)
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

    args.bignore = param["anyBaseOCR"]["deskew"]["bignore"]
    args.escale = param["anyBaseOCR"]["deskew"]["escale"]
    args.threshold = param["anyBaseOCR"]["deskew"]["threshold"]
    args.lo = param["anyBaseOCR"]["deskew"]["lo"]
    args.hi = param["anyBaseOCR"]["deskew"]["hi"]
    args.maxskew = param["anyBaseOCR"]["deskew"]["maxskew"]
    args.skewsteps = param["anyBaseOCR"]["deskew"]["skewsteps"]
    args.debug = param["anyBaseOCR"]["deskew"]["debug"]
    args.parallel = param["anyBaseOCR"]["deskew"]["parallel"]
    ### End to read parameters

    # mendatory parameter check
    if not args.mets or not args.Input or not args.Output or not args.work:
        parser.print_help()
        print("Example: python ocrd-anyBaseOCR-deskew.py -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)")
        sys.exit(0)

    if args.work:
        if not os.path.exists(args.work):
            os.mkdir(args.work)

    files = parseXML(args.mets)
    fname=[]
    for i, f in enumerate(files):
        fname.append(deskew(str(f),i+1))
    write_to_xml(fname)
