(TeX-add-style-hook
 "iclr2019_conference"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("graphicx" "dvips" "pdftex")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "math_commands"
    "article"
    "art10"
    "times"
    "hyperref"
    "url"
    "graphicx")
   (TeX-add-symbols
    "fix"
    "new"
    "arraystretch")
   (LaTeX-add-labels
    "loss1"
    "reparam"
    "loss2"
    "gen_inst"
    "headings"
    "others"
    "sample-table")
   (LaTeX-add-bibliographies))
 :latex)

