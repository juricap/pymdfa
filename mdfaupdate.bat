c:\Python27\Scripts\ipython.exe nbconvert --to html mdfaintro.ipynb
c:\Python27\Scripts\ipython.exe nbconvert --to html mdfafast.ipynb
c:\cygwin\bin\ssh.exe juricap@www mkdir /home/juricap/mdfa
c:\cygwin\bin\scp.exe -r mdfa* juricap@www:/home/juricap/mdfa/
c:\cygwin\bin\scp.exe pymdfa*.py juricap@www:/home/juricap/mdfa/
c:\cygwin\bin\ssh.exe juricap@www chmod -R a+r /home/juricap/mdfa/*

