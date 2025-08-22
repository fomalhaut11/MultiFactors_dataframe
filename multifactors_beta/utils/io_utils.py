#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O utilities to fix encoding issues on Windows
"""
import sys
import io
import codecs


def fix_io_encoding():
    """
    Fix I/O encoding issues on Windows by setting UTF-8 encoding
    """
    # Set UTF-8 encoding for stdout and stderr
    if sys.platform == 'win32':
        # For Windows, we need to handle encoding properly
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        
        # Also set the console code page to UTF-8
        try:
            import subprocess
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
        except:
            pass
    
    # Set default encoding for open()
    if hasattr(sys, 'setdefaultencoding'):
        sys.setdefaultencoding('utf-8')


def safe_print(*args, **kwargs):
    """
    Safe print function that handles encoding issues
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Convert all arguments to ASCII-safe strings
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                # Replace non-ASCII characters with their unicode escape
                safe_args.append(arg.encode('ascii', 'backslashreplace').decode('ascii'))
            else:
                safe_args.append(str(arg))
        print(*safe_args, **kwargs)
    except Exception as e:
        # Last resort: print to a file
        with open('output_log.txt', 'a', encoding='utf-8') as f:
            print(*args, file=f, **kwargs)
            print(f"[Error writing to console: {e}]", file=f)


class SafeIOWrapper:
    """
    Context manager for safe I/O operations
    """
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create safe wrappers
        sys.stdout = io.TextIOWrapper(
            io.BytesIO(), 
            encoding='utf-8', 
            errors='replace',
            line_buffering=True
        )
        sys.stderr = io.TextIOWrapper(
            io.BytesIO(), 
            encoding='utf-8', 
            errors='replace',
            line_buffering=True
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        return False


def run_with_safe_io(func, *args, **kwargs):
    """
    Run a function with safe I/O handling
    """
    fix_io_encoding()
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # If there's still an I/O error, run with redirected output
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
            temp_file = f.name
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            try:
                sys.stdout = f
                sys.stderr = f
                result = func(*args, **kwargs)
                return result
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                # Print the output
                if os.path.exists(temp_file):
                    with open(temp_file, 'r', encoding='utf-8') as rf:
                        content = rf.read()
                        if content:
                            print("Output saved to temp file due to I/O issues:")
                            print(content)
                    os.unlink(temp_file)