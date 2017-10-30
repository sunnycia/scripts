#coding:gbk 
class Colored(object):  
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    FUCHSIA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'      
  
    #: no color  
    RESET = '\033[0m'
  
    def color_str(self, color, s):  
        return '{}{}{}'.format(  
            getattr(self, color),  
            s,  
            self.RESET  
        )  
  
    def red(self, s):  
        return self.color_str('RED', s)  
  
    def green(self, s):  
        return self.color_str('GREEN', s)  
  
    def yellow(self, s):  
        return self.color_str('YELLOW', s)  
  
    def blue(self, s):  
        return self.color_str('BLUE', s)  
  
    def fuchsia(self, s):  
        return self.color_str('FUCHSIA', s)  
  
    def cyan(self, s):  
        return self.color_str('CYAN', s)  
  
    def white(self, s):  
        return self.color_str('WHITE', s)  