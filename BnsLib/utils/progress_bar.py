from datetime import datetime, timedelta
import sys
import multiprocessing as mp

class progress_tracker():
    """A class that implements and prints a dynamic progress bar to
    stdout.
    
    Arguments
    ---------
    num_of_steps : int
        The number of iterations that is expected to occur.
    name : {str, 'Progress'}
        The name for the header of the progress bar. It will be followed
        by a colon ':' when printed.
    steps_taken : {int, 0}
        The number of steps that are already completed.
    """
    def __init__(self, num_of_steps, name='Progress', steps_taken=0):
        self.t_start = datetime.now()
        self.num_of_steps = num_of_steps
        self.steps_taken = steps_taken
        self.name = name
        self._printed_header = False
        self.last_string_length = 0
    
    def __len__(self):
        return self.num_of_steps
    
    @property
    def eta(self):
        now = datetime.now()
        return(int(round(float((now - self.t_start).seconds) / float(self.steps_taken) * float(self.num_of_steps - self.steps_taken))))
    
    @property
    def percentage(self):
        return(int(100 * float(self.steps_taken) / float(self.num_of_steps)))
    
    def get_print_string(self):
        curr_perc = self.percentage
        real_perc = self.percentage
        #Length of the progress bar is 25. Hence one step equates to 4%.
        bar_len = 25
        
        if not curr_perc % 4 == 0:
            curr_perc -= curr_perc % 4
        
        if int(curr_perc / 4) > 0:
            s = '[' + '=' * (int(curr_perc / 4) - 1) + '>' + '.' * (bar_len - int(curr_perc / 4)) + ']'
        else:
            s = '[' + '.' * bar_len + ']'
        
        tot_str = str(self.num_of_steps)
        curr_str = str(self.steps_taken)
        curr_str = ' ' * (len(tot_str) - len(curr_str)) + curr_str
        eta = str(timedelta(seconds=self.eta)) + 's'
        perc_str = ' ' * (len('100') - len(str(real_perc))) + str(real_perc)
        
        out_str = curr_str + '/' + tot_str + ': ' + s + ' ' + perc_str + '%' + ' ETA: ' + eta
        
        if self.last_string_length > len(out_str):
            back = '\b \b' * (self.last_string_length - len(out_str))
        else:
            back = ''
        
        #back = '\b \b' * self.last_string_length
        
        self.last_string_length = len(out_str)
        
        return(back + '\r' + out_str)
        #return(back + out_str)
    
    def print_progress_bar(self, update=True):
        if not self._printed_header:
            print(self.name + ':')
            self._printed_header = True
        
        if update:
            sys.stdout.write(self.get_print_string())
            sys.stdout.flush()
            if self.steps_taken == self.num_of_steps:
                self.print_final(update=update)
        else:
            print(self.get_print_string())
            if self.steps_taken == self.num_of_steps:
                self.print_final(update=update)
    
    def iterate(self, iterate_by=1, print_prog_bar=True, update=True):
        if iterate_by > 0:
            self.steps_taken += iterate_by
            if print_prog_bar:
                self.print_progress_bar(update=update)
    
    def print_final(self, update=True):
        final_str = str(self.steps_taken) + '/' + str(self.num_of_steps) + ': [' + 25 * '=' + '] 100% - Time elapsed: ' + str(timedelta(seconds=(datetime.now() - self.t_start).seconds)) + 's'
        if update:
            clear_str = '\b \b' * self.last_string_length
            
            sys.stdout.write(clear_str + final_str + '\n')
            sys.stdout.flush()
        else:
            print(final_str)

class mp_progress_tracker(progress_tracker):
    """A class that implements and prints a dynamic progress bar to
    stdout. This special case is multiprocessing save.
    
    Arguments
    ---------
    num_of_steps : int
        The number of iterations that is expected to occur.
    name : {str, 'Progress'}
        The name for the header of the progress bar. It will be followed
        by a colon ':' when printed.
    steps_taken : {int, 0}
        The number of steps that are already completed.
    """
    def __init__(self, num_of_steps, name='Progress', steps_taken=0):
        self._printed_header_val = mp.Value('i', False)
        self.last_string_length_val = mp.Value('i', 0)
        super().__init__(num_of_steps, name=name,
                         steps_taken=steps_taken)
        self.steps_taken = mp.Value('i', steps_taken)
    
    @property
    def _printed_header(self):
        return bool(self._printed_header_val.value)
    
    @_printed_header.setter
    def _printed_header(self, boolean):
        with self._printed_header_val.get_lock():
            self._printed_header_val.value = int(boolean)
    
    @property
    def last_string_length(self):
        return self.last_string_length_val.value
    
    @last_string_length.setter
    def last_string_length(self, length):
        with self.last_string_length_val.get_lock():
            self.last_string_length_val.value = length
    
    @property
    def eta(self):
        now = datetime.now()
        return(int(round(float((now - self.t_start).seconds) / float(self.steps_taken.value) * float(self.num_of_steps - self.steps_taken.value))))
    
    @property
    def percentage(self):
        return(int(100 * float(self.steps_taken.value) / float(self.num_of_steps)))
    
    def get_print_string(self):
        curr_perc = self.percentage
        real_perc = self.percentage
        #Length of the progress bar is 25. Hence one step equates to 4%.
        bar_len = 25
        
        if not curr_perc % 4 == 0:
            curr_perc -= curr_perc % 4
        
        if int(curr_perc / 4) > 0:
            s = '[' + '=' * (int(curr_perc / 4) - 1) + '>' + '.' * (bar_len - int(curr_perc / 4)) + ']'
        else:
            s = '[' + '.' * bar_len + ']'
        
        tot_str = str(self.num_of_steps)
        curr_str = str(self.steps_taken.value)
        curr_str = ' ' * (len(tot_str) - len(curr_str)) + curr_str
        eta = str(timedelta(seconds=self.eta)) + 's'
        perc_str = ' ' * (len('100') - len(str(real_perc))) + str(real_perc)
        
        out_str = curr_str + '/' + tot_str + ': ' + s + ' ' + perc_str + '%' + ' ETA: ' + eta
        
        if self.last_string_length > len(out_str):
            back = '\b \b' * (self.last_string_length - len(out_str))
        else:
            back = ''
        
        #back = '\b \b' * self.last_string_length
        
        self.last_string_length = len(out_str)
        
        return(back + '\r' + out_str)
        #return(back + out_str)
    
    def print_progress_bar(self, update=True):
        if not self._printed_header:
            print(self.name + ':')
            self._printed_header = True
        
        if update:
            sys.stdout.write(self.get_print_string())
            sys.stdout.flush()
            if self.steps_taken.value == self.num_of_steps:
                self.print_final(update=update)
        else:
            print(self.get_print_string())
            if self.steps_taken.value == self.num_of_steps:
                self.print_final(update=update)
    
    def iterate(self, iterate_by=1, print_prog_bar=True, update=True):
        with self.steps_taken.get_lock():
            if iterate_by > 0:
                self.steps_taken.value += iterate_by
                if print_prog_bar:
                    self.print_progress_bar(update=update)
    
    def print_final(self, update=True):
        final_str = str(self.steps_taken.value) + '/' + str(self.num_of_steps) + ': [' + 25 * '=' + '] 100% - Time elapsed: ' + str(timedelta(seconds=(datetime.now() - self.t_start).seconds)) + 's'
        if update:
            clear_str = '\b \b' * self.last_string_length
            
            sys.stdout.write(clear_str + final_str + '\n')
            sys.stdout.flush()
        else:
            print(final_str)
