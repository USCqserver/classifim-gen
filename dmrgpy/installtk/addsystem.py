#!/usr/bin/python
        
# SPDX-License-Identifier: GPL-3.0-or-later

import os

# different files for Linux and Mac
import platform


def addrc():
    """
    Add program to the .bashrc
    """
    shell = os.environ["SHELL"]
    if "bash" in shell: shellrc = ".bashrc"
    elif "zsh" in shell: shellrc = ".zshrc"
    else:
        print("Unrecognise shell",name)
        exit()
    dmrg_installtk_dir = os.path.dirname(os.path.realpath(__file__))
    dmrg_src_dir = os.path.realpath(
        os.path.join(dmrg_installtk_dir, os.pardir, "src"))
    if platform.system()=="Linux":
      shellrc = os.environ["HOME"]+"/"+shellrc # path to .bashrc
      print("Detected Linux system")
      f = open(shellrc).read() # read bashrc
    else:
      # Michael's fix for Mac
      f = ''
      for profileString in ["/.bash_profile", "/.bash_login","/.profile"]:
        bashrc = os.environ["HOME"]+profileString # path to .bashrc
        try:
          print(profileString)
          f = open(bashrc).read() # read bashrc
          break
        except FileNotFoundError:
          pass
      print("Detected Mac system")

    if "export DMRGROOT=" in f:
        print("DMRGROOT already in your ",shellrc)
        return

    print("Adding DMRGROOT to your ",shellrc)

    route = "\n\n###############################\n"
    route += "  export DMRGROOT=\"" + dmrg_src_dir + "\"\n"
    route += "###############################\n"

    open(shellrc,"w").write(f+route) # write in the bashrc

    print("Added \n"+route+"\n route to ",shellrc)


if __name__=="__main__":
    addbashrc() # add to the .bashrc

