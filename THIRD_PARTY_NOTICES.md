# Third-party notices

MedMLX-authored additions are licensed under [Apache-2.0](LICENSE).
The upstream attribution and license notices are retained below.

synthseg-mlx includes portions of SynthSeg and its ext/lab2im utilities.

Copyright 2020 Benjamin Billot
Original upstream: https://github.com/BBillot/SynthSeg
lab2im upstream: https://github.com/BBillot/lab2im
Licensed under the Apache License, Version 2.0; see THIRD_PARTY_NOTICES.md.

Source: the SynthSeg fork at revision
d6a64db19ec7b8b71c64edd59e7e485e0235c26c.
Copied numerical code: SynthSeg/{predict_synthseg,labels}.py and
ext/lab2im/{utils,edit_volumes}.py. Bundled data: the seven SynthSeg 2.0
label, topology and name arrays listed in synthseg_mlx/label_data.py.

Changes: extracted the required NumPy/SciPy functions, adapted imports and
package layout, moved image I/O to caller-owned array inputs and an optional
resampled-image hook, removed framework tensor adapters, and made the
orientation and label helpers' one-element integer assignments explicit for NumPy 2.
The MedMLX runner also uses the copied NIfTI and CSV helpers (load_volume,
save_volume, get_volume_info, get_list_labels without its label-folder branch,
mkdir, write_csv).
Original
copyright and citation headers are retained in the copied Python files.

If you use this code, please cite the SynthSeg papers identified at
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib and
https://github.com/BBillot/lab2im/blob/master/bibtex.bib.

The extracted numerical files listed above retain their upstream Apache-2.0
headers. The label arrays come from the same source repository and revision,
whose top-level `LICENSE.txt` is Apache-2.0.

## Network and reference provenance

The native MLX network and array engine were moved from RadNN `cbaa1ac` in
synthseg-mlx commit `333f57dcd5dc821231e922b466d0d9755c1e8768`. The MLX model
first appears in RadNN commit `b35a3ac0eda371978bb1481bc8663a8e0bcc8393`.
The native graph uses MLX classes and operators, retaining the SynthSeg network
structure, checkpoint layer names and operation order described in
`docs/technical.md`.

The independent reference recorder `scripts/make_variant_fixtures.py` executes
selected definitions from the external SynthSeg reference revision
`2a2aa3bbfccb83f8253a51ca8b329b9938a2646d`, including:

- `ext/neuron/models.py`: the `unet`, `conv_enc` and `conv_dec` builders; its
  file header states `License: GPLv3`.
- `ext/neuron/utils.py`: coordinate helpers used by the reference graph; its
  file header also states `License: GPLv3`.

These file-specific headers differ from that reference repository's top-level
Apache-2.0 `LICENSE.txt`. The external neuron files are not vendored in this
checkout or included in its wheel or source archive. The reference recorder
executes them from an explicitly selected source checkout; its numerical source
revision remains unchanged.

### Historical Neurite attribution

Copyright 2020 Adrian V. Dalca

The historical Neurite source provides an explicit Apache-2.0 grant for relevant
network builders and coordinate helpers. Adrian Dalca's commit
`b8efac0fcfceec8017c324b3b4de61c16b6903f2` added the Apache-2.0 license;
commit `c516287431d37ba6f2e0f5fcfc69d87cda552f5e` changed the notices in
`neurite/tf/models.py` and `neurite/tf/utils/utils.py` from GPLv3 to Apache-2.0.
See the full Apache-2.0 license below.

The `unet`, `conv_enc`, `conv_dec`, `volshape_to_meshgrid` and `meshgrid` function
bodies are unchanged by that file-notice commit, apart from documentation. The
two coordinate helpers have identical function bodies and signatures in the
SynthSeg reference revision and this Apache-headered Neurite revision. The three
builders share network-construction code but are not identical: the SynthSeg
fork changes input composition, dropout handling and decoder skip selection.
The native MLX graph retains the fixed network schedules and checkpoint layer
names; the Apache-headered `SynthSeg/predict_synthseg.py` also declares the
standard, robust, parcellation and QC architectures, including the denoiser's
omitted skip connections. The native QC crop and Gaussian coefficient generation
follow the Apache-headered `MakeShape` and `ext/lab2im/edit_tensors.py` helpers,
with local MLX array and Metal implementations.

This provenance description distinguishes the extracted numerical files,
external reference builders and native graph implementation. It does not make
a determination about copied or translated source expression, independent
implementation of the architecture, or the license scope of the native graph.

Pinned source headers:

- https://github.com/BBillot/SynthSeg/blob/2a2aa3bbfccb83f8253a51ca8b329b9938a2646d/ext/neuron/models.py#L10
- https://github.com/BBillot/SynthSeg/blob/2a2aa3bbfccb83f8253a51ca8b329b9938a2646d/ext/neuron/utils.py#L16
- https://github.com/BBillot/SynthSeg/blob/2a2aa3bbfccb83f8253a51ca8b329b9938a2646d/LICENSE.txt

Historical Apache sources:

- https://github.com/adalca/neurite/commit/b8efac0fcfceec8017c324b3b4de61c16b6903f2
- https://github.com/adalca/neurite/commit/c516287431d37ba6f2e0f5fcfc69d87cda552f5e
- https://github.com/adalca/neurite/blob/c516287431d37ba6f2e0f5fcfc69d87cda552f5e/neurite/tf/models.py
- https://github.com/adalca/neurite/blob/c516287431d37ba6f2e0f5fcfc69d87cda552f5e/neurite/tf/utils/utils.py

Dependencies declared in [pyproject.toml](pyproject.toml) are installed separately
and retain their respective licenses.


## Full license: LICENSE_SynthSeg.txt

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
