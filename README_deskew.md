# README file for Skew Correction component

Filename : ocrd-anyBaseOCR-deskew.py

Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
Note:
1. this work has been done in DFKI, Kaiserslautern, Germany.
2. The parameters values are read from ocrd-anyBaseOCR-parameter.json file. The values can be changed in that file.
3. The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/).
4. The sequence of operations is: binarization, deskewing, cropping and dewarping (or can also be: binarization, dewarping, deskewing, and cropping; depends upon use-case).
5. Sample files are available at [OCR-D/assets](https://github.com/OCR-D/ocrd-assets/tree/master/data/dfki-testdata)

# Method Behaviour

This function takes a document image as input and do the skew correction of that document.

# LICENSE

 License: ocropus-nlbin.py (from https://github.com/tmbdev/ocropy/) contains both functionalities: binarization and skew correction. This method (ocrd-anyBaseOCR-deskew.py) only contains the skew correction functionality of ocropus-nlbin.py. It still has the same licenses as ocropus-nlbin, i.e Apache 2.0 (the ocropy license details are pasted below).
 This file is dependend on ocrolib library which comes from https://github.com/tmbdev/ocropy/.

```
Copyright 2014 Thomas M. Breuel

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
 limitations under the License.
```

# Usage:

```sh
python ocrd-anyBaseOCR-deskew.py -m (path to met input file) -I (Input group name) -O (Output group name) -w (Working directory)
	[-p (path to parameter file) -o (METs output filename)]
```

# Example:

```sh
# git clone https://github.com/OCR-D/assets
cd assets/data/dfki-testdata/data/
python ocrd-anyBaseOCR-binarize.py \
  -m mets.xml \
  -I OCR-D-IMG-BIN \
  -O OCR-D-IMG-DESKEW-VERIFY
diff OCR-D-IMG-DESKEW/becker_quaestio_1586_00013.ds OCR-DIMG-BIN-DESKEW-VERIFY/becker_quaestio_1586_00013.ds
# should have zero return code
```
