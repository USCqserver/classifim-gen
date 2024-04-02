import datetime


class ProgressPrinter:
    """
    A small utility class for printing progress in an IPython notebook.

    Example:
    >>> pp = ProgressPrinter()
    >>> pp.pprintln(pp.get_now_str())
    >>> for i in range(200):
    >>>     if i == 29:
    >>>         # '\r' ensures a new line is started.
    >>>         pp.pprint('\r[i == 29] <Your message>')
    >>>     # Prints '.' to indicate progress.
    >>>     # Every 10th iteration prints the iteration number.
    >>>     # Every 100th iteration prints a newline.
    >>>     pp.inc_i()
    >>> # Use '\r' to ensure a new line is started.
    >>> pp.pprintln('\r' + pp.get_now_str())
    """
    def __init__(self, new_line=True):
        self.print_i = 0
        self.new_line = new_line
        
    def ensure_new_line(self):
        """
        Prints a newline character unless we are already at the beginning of the line.
        
        Note:
            This is called automatically when the arg to pprint or pprintln
            starts with '\r'.
        """
        if not self.new_line:
            self.pprintln()
    
    @staticmethod
    def _args_asks_for_new_line(args):
        if len(args) == 0:
            return False
        arg = args[0]
        if isinstance(arg, str):
            return arg.startswith('\r')
        if isinstance(arg, int):
            return False
        raise NotImplementedError(f"_args_asks_for_new_line for type {type(arg)} of {arg}")

    def pprint(self, *args):
        if self._args_asks_for_new_line(args):
            self.ensure_new_line()
        print(*args, end="")
        self.new_line = False
        
    def pprintln(self, *args):
        if self._args_asks_for_new_line(args):
            self.ensure_new_line()
        print(*args, end="\n")
        self.new_line = True
        
    def inc_i(self, char='.'):
        self.print_i += 1
        self.pprint(char)
        
        if self.print_i % 100 == 0 and not self.new_line:
            self.pprintln()
        elif self.print_i % 10 == 0:
            self.pprint(self.print_i)
    
    @staticmethod
    def get_now_str():
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        return formatted_time
