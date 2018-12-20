(TeX-add-style-hook
 "iclr2019_conference"
 (lambda ()
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "math_commands"
    "article"
    "art10"
    "times"
    "hyperref"
    "url")
   (TeX-add-symbols
    "fix"
    "new")
   (LaTeX-add-labels
    "loss1"
    "reparam"
    "loss2")
   (LaTeX-add-bibliographies))
 :latex)

