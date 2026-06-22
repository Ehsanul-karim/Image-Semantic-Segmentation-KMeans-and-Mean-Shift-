all:
	pdflatex -interaction=nonstopmode main.tex
	bibtex main || true
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

preview:
	sed 's/\\includeallresultgallerytrue/\\includeallresultgalleryfalse/' main.tex > main_quick_preview.tex
	pdflatex -interaction=nonstopmode main_quick_preview.tex
	bibtex main_quick_preview || true
	pdflatex -interaction=nonstopmode main_quick_preview.tex
	pdflatex -interaction=nonstopmode main_quick_preview.tex
