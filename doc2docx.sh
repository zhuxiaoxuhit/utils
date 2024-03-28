ls *.doc | xargs -I {} libreoffice --invisible -convert-to docx:"MS Word 2007 XML" {}
