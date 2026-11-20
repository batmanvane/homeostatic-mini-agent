note.pdf: Makefile note.md fig_setup.png fig_shadow.png fig_steering_compact.png
	pandoc note.md -o note.pdf --pdf-engine=xelatex -V geometry:margin=1.2cm -V fontsize=10pt -V mainfont="Charter" -V mathfont="STIX Two Math" -V linkcolor=black -V linestretch=0.95 --resource-path=.
