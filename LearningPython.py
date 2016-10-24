# Python 2.7 CheatSheet
# http://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000

# input/output
# escape
s = '\'\\'
print s
# original
s = r'\'\\'
print s
# multi-line
print '''this
... is
... a
... test'''

# multi-part
print 'this', 'is', 'a', 'test'

# encoding
# ASCII 127 codes
# ASCII is not enough for other languages
# Unicode normally uses two bytes for a character and is supported natively
# but Unicode is very inefficient and redundent under certain circumstances
# UTF-8 variable length, 1~6 bytes for a character, normally each Chinese character 
# takes 3 bytes
# in computer memory, Unicode is universarly applied; when storing info into the disk
# they are transformed into UTF-8
# in Python u'...' means Unicode
print u'中文'
# from Unicode to UTF-8
print u'ABC'.encode('utf-8')
print u'中文'.encode('utf-8')
# by default Python display only ASCII
# tell the Python Interpreter to read by UTF-8

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# elements of tuples are immutable
# that means if regarding a tuple as 
# a ordered collection of pointers
# each pointer is const

t = (1, 2, [2, 3])
# t[0] = 2
# t[1] = 1
t[2][0] = 10

# for a tuple consisting of only one element
t = (1, )

# immutable elements' methods do not modify themselves
# set and dict only accept immutable elements as keys

# function
# return None == return (void)
# procedures (functions without return) return None

# pass is a place holder to prevent error when running the code
# but actually does nothing

def void():
    pass

# multiple returns of a function actually returns a tuple
def test():
    return 1, 2
t = test()
print t

# exception
def my_abs(x):
    if not isinstance(x, (int, float)):
        raise TypeError('incorrect type')
    if x >= 0:
        return x
    else:
        return -x

# default parameter
# default parameter must point to immutables
# otherwise
def add_end(L=[]):
    L.append("End")
    return L
# None could be used as a default
# note that mutable params will change after the routine
def add_end(L=None):
    if L is None:
        return []
    else:
        return L.append("End")
# * and **
def ast(*numbers):
    sum = 0
    for number in numbers:
        sum += number
    return sum
# now a tuple is read in
# could also be None
# if one-element tuple
# output "(a,)"
ast(1, 2, 3, 4, 5)
ast()
# similarly
a = [1, 3, 5]
ast(*a)

def key_fun(x, y, **kw):
    print x, y, kw
key_fun(1, 2)
key_fun(1, 2, city="LA", state="CA")
d = {"city": "LA", "state": "CA"}
key_fun(1, 2, **d)

# combination order
# def order(a, b, c=0, *k, **kw)
# note that if order(1, 2, 3, 'a', 'b', time="now")
# or
# t = [1, 2, 3, 4]
# d = {"city": "LA"}
# order(*t, **d)

# slicing
L = range(100)
print L[:10]
print L[-10:]
print L[1:10]

# every two elements from the first 10
print L[:10:2]

# every 5 elements
print L[::5]

# duplicate
print L[:]

# reverse
print L[::-1]

# for iteration over iterable
from collections import Iterable
isinstance('avc', Iterable)

d = {'1': 1}
for key, value in d.iteritems():
    print key, value
    
# enumerate
for index, value in enumerate([1, 2, 3]):
    print index, value
    
# list comprehension
[x * x for x in range(1, 11)]
[x * x for x in range(1, 11) if x % 2 == 0]
[m + n for m in 'abc' for n in 'ABC']

# similarly
x = 'a' if y == 'c' else 'b'

# generator

L = [x * x for x in range(10)]
print L
g = (x * x for x in range(10))
print g

print g.next()

for n in g:
    print n

def fib(max_num):
    n, a, b = 0, 0, 1
    while n < max_num:
        yield b
        a, b = b, a + b
        n = n + 1

for n in fib(6):
    print n
    
# note that yield is like a breaking point with return
# when you call next() it resumes

# function name is actually a pointer
# actually everything in Python is a pointer
print abs(-10)
abs = 10
print abs(-10)

# recover Python interpreter now!

# higher-order function
def add(x, y, f):
    return f(x) + f(y)

add(-10, -5, abs)

# map
# map() takes two arguments, one function and another sequence
# the function is applied on every element in the sequence

def f(x):
    return x * x

print map(f, range(5))

# reduce
# reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)

def bar(x, y):
    return x * 10 + y

print reduce(bar, [1, 2, 3, 4, 5])

def str2int(s):
    def fn(x, y):
        return x * 10 + y
    def char2num(s):
        return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s]
    return reduce(fn, map(char2num, s))
    
# or lambda x, y: x * 10 + y

# filter
# filter applies to every element but it returns true/false

def is_odd(x):
    return n % 2 == 0

print filter(is_odd, range(10))

# sorted
# by default x ? y -> return 1 if x > y 0 if x == y -1 if x < y 

def reversed_order(x, y):
    if x > y:
        return -1
    if x < y:
        return 1
    return 0
    
print sorted([36, 5, 12, 9, 21], reversed_order)
   
# closure
# internal function can access variables in the higher scope

def lazy_sum(*args):
    def sum():
        ax = 0
        for n in args:
            ax += n
        return ax
    return sum

f1 = lazy_sum(1, 2, 3, 4, 5)
f2 = lazy_sum(2, 3, 4, 5, 6)
print f1()
print f2()
print f1 == f2

# Do not access variables that would be changed latter in the same closure
def count():
    fs = []
    for i in range(1, 4):
         def f():
            return i * i
         fs.append(f)
    return fs
f1, f2, f3 = fs

# functions reference the same i
print f1(), f2(), f3()

# lambda
# lambda expression is restricted to one line and one expression
# variables before : are arguments
# no need to write return
# lambda can have no arguments
g = lambda x, y: (x + y, x - y)

# decorator
# every function variable has an attribute __name__
def now():
    print "2016-08-24"
f = now

print now.__name__, f.__name__

# decorator can dynamically expands the usage of a function
def log(func):
    def wrapper(*args, **kw):
        print "calling %s()" % func.__name__
        return func(*args, **kw)
    return wrapper

@log
def now():
    print "2016-08-24"

# this definition is equivalent to 
# now = log(now)

now()
print now.__name__

# decorator with parameters
def log(text):
    def decorator(func):
        def wrapper(*args, **kw):
            print "%s %s()" % (text, func.__name__)
            return func(*args, **kw)
        return wrapper
    return decorator

@log("executing")
def now():
    print "2016-08-24"

now()
print now.__name__
# this time
# now = log(text)(now)

# note that here the signature of the function is changed
# this might cause some problems
# therefore
# add @functools.wraps(func) in front of the wrapper function

import functools

def log(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            print "%s %s()" % (text, func.__name__)
            return func(*args, **kw)
        return wrapper
    return decorator

@log("executing")
def now():
    print "2016-08-24"

now() 
print now.__name__

# module
# each .py file is a module
# different modules do not share the same name space
# so there are not conflincts in naming among different modules
# higher level of name space than module is package (a folder)
# each package must have a __init__.py module (could be empty)
# __init__.py's module name is the name of the package
# could be hierarchical packaging

'''example hello.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"A Test Module" (__doc__)

__author__ = "Welly Zhang"

import sys

def test():
    args = sys.argv
    if len(args) == 1:
        print "Hello, World!"
    elif len(args) == 2:
        print "Hello, %s!" % args[1]
    else:
        print "Too many arguments!"

if __name__ == "__main__":
    test()
'''

# the if condition makes sure that the function is called 
# only when the module is run rather than imported

# import alias
try:
    import cStringIO as StringIO
except:
    import StringIO

# scope
# normal naming convention defaults to public
# __*__ is also public but has special purposes
# _* or __* should be treated as private
# but could still be accessed publicly if you wish

# third party package
# python uses setuptools to manage packages
# esay_install and pip are two mature command line software for package management
# module search paths are stored in sys.path
# could be changed by editing PYTHONPATH

# __future__
# for compatibility of Python 2.7 to Python 3.x

'''
x = 10 / 3
print x

from __future__ import division

x = 10 / 3
print x
'''

# class
# note that object is the parent class of student

class Student(object):
    
    def __init__(self, name, score):
        self.name = name
        self.score = score
    
    def print_score(self):
        print "%s %d" %(self.name, self.score)
        
lisa = Student("Lisa", 50)
lisa.print_score()

richard = Student("Richard", 60)
richard.age = 12
print richard.age, richard.name, richard.score
# print lisa.age

# private attribute

class Student(object):
    
    def __init__(self, name, score):
        self.__name = name
        self.__score = score
    
    def print_score(self):
        print "%s %s" % (self.__name, self.__score)
        
lisa = Student("Lisa", 50)
# print lisa.__name
# print lisa._Student__name

# _* should be treated as private but could be accessed publicly

# inheritance and polymorphism

# a subclass instance is of both its super class and the class itself

# type()
print type(123)
print type("hello")
print type(None)
print type(abs)

import types
print type('abc') == types.StringType
print type(str) == types.TypeType

# isinstance()
print isinstance('abc', str)
print isinstance('abc', (unicode, str))

# dir()
# prints all attributes and methods of an instance

print dir("ABC")
print "ABC".__len__()

# build-in method len() actually calls the __len__ method in the class

# getattr(), setattr(), hasattr()

class MyObject(object):
    
    def __init__(self):
        self.x = 2
    
    def power(self):
        return self.x * self.x
        
obj = MyObject()
print hasattr(obj, 'x')
print hasattr(obj, 'y')
setattr(obj, 'y', 4)
print getattr(obj, 'y')
print obj.y

print hasattr(obj, 'power')
fn = getattr(obj, 'power')
print fn()

# multi-inheritance
# Mixin
# class MyTCPServer(TCPServer, ForkingMixin)

# __*__ methods
class Student(object):
    
    def __init__(self, name):
        self.name = name
    
    def __iter__(self):
        return self
    
    def __str__(self):
        return "Student oject (name=%s)" % self.name
    
#   __repr__ == __str__
    
lisa = Student("Lisa")
print lisa
# lisa

# exception
# note the execution logic of the following code snippet
# finally is optional
# could also have multiple excepts

try:
    print "trying"
    r = 10 / 0
    print "result:", r
except ZeroDivisionError, e:
    print "except:", e
else:
    print "else"
finally:
    print "finally"
print "end"


try:
    print "trying"
    r = 10 / 2
    print "result:", r
except ZeroDivisionError, e:
    print "except:", e
else:
    print "else"
finally:
    print "finally"
print "end"

# all exceptions extend BaseException
# and the exception in except includes all subclasses
# of the exception; exception is caught only once
# exception will be thrown to a higher level if it's not caught

# Stack trace and logging
# logging module records the exceptions as log
# and allows the program to continue after the exception

# raise

class FooError(StandardError):
    pass
    
def foo(s):
    n = int(s)
    if n == 0:
        raise FooError("Invalid input")
    return 10 / n
    
# another way of recording the error
# if raise does not have a argument, the error is 
# thrown as it is

def foo(s):
    n = int(s)
    return 10 / n

def bar(s):
    try:
        return foo(s) * 2
    except StandardError, e:
        print "Error"
        raise
        
# debug

# assert
# if the expression is false, the sentence would be printed
# it's actually the AssertionError

n = 0
assert n != 0, "n is zero"

# pdb
# python -m pdb file.py
# l - review the code
# n - single step
# p var - check value
# q - quit

# import pdb
# pdb.set_trace() sets the breaking point
# c - continue

# module test
# unitest and doctest
# unitest is encouraged to be used in command line
# write a test class
# doctest tests the expressions and output written in the comments
# ... to ommit
# save this to another file

def abs(n):
    """
    Function to get absolute value.
    
    Example:
    
    >>> abs(1)
    1
    >>> abs(-1)
    1
    >>> abs(0)
    0
    >>> abs(None)
    Traceback (most recent call last):
        ...
    TypeError: bad operand type for abs(): 'NoneType'
    """
    
    return n if n >= 0 else (-n)

if __name__ == "__main__":
    
    import doctest
    doctest.testmod()
    
# I/O
# Synchronous and Asynchronous
# Sync: CPU waits until data is ready
# Async: CPU does others until data is ready 
#   but need other methods to be notified 

# file-like object
# has read()
# file-like objects could be files and streams

# to read binary files or files not encoded with ASCII, use "rb"
# Example
# >>> f = open("gbk.txt", "rb")
# >>> u = f.read.decode("gbk")
# or 
# >>> import codecs
# >>> with codecs.open("gbk.txt", "r", "gbk") as f:
# ...     f.read()

# OS
# os.name
# os.uname()
# os.environ 
# os.getenv("PATH")
# os.path.join() and os.path.split() correctly handle the delimiters
# os.path.splitext() gets the file extension

# pickling and unpickling
# cPickle faster and pickle slower

try:
    import cPickle as pickle
except ImportError:
    import pickle

d = dict(name="bob", age=32, score = 100) 
print pickle.dumps(d)
with open("test.txt", "wb") as f:
    pickle.dump(d,f)
with open("test.txt", "rb") as f:
    d = pickle.load(f)
    
# pickling is for Python only
# suggest using JSON

# import json
# json.dumps() and json.dump()
# json.loads() and json.load()
# for class

import json

class Student(object):
    
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score
        
s = Student("Bob", 21, 100)

def student2dict(std):
    d = {"name": std.name, "age": std.age, "score": std.score}
    return d
    
print json.dumps(s, default=student2dict)
# json.dumps(s, default=lambda obj: obj.__dict__)
# the __dict__ method stores the attributes

def dict2student(d):
    return Student(d.name, d.age, d.score)
    
json.loads(json_str, object_hook=dict2student)

# Process and Thread
# Process is a individual program
# A process could have many threads that do different tasks in the meantime
# Each process has at least one thread

# Multiprocessing
# fork replicates a child process that shares the same code
# child process always returns 0
# parent process returns its pid

import os
pid = os.getpid()
print pid
pid = os.fork()
if pid == 0:
    print "I am a child process %s and my parent is %s" % (os.getppid(),
                                                           os.getpid())
else:
    print "I just created a child process %s" % os.getpid()

# portable: Windows does not have fork
# multiprocessing
# the join method waits until the process ends, used for synchronization

from multiprocessing import Process
import os

def run_proc(name):
    print "Running %s pid %s" % (name, os.getpid())

print "parent process %s" % os.getpid()
p = Process(target=run_proc, args=("Test", ))
print "Process starts"
p.start()
p.join()
print "Process ends"

# pool
# must close before join and no more processes after close
# pool has size, by defaul the number of cores in the CPU


from multiprocessing import Pool
import os, time, random

def long_time_task(name):
    print "Run task %s pid %s" % (name, os.getpid())
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print "Task %s runs for %0.2f seconds" % (name, (end - start))

print "Parent process %s" % os.getpid()
p = Pool()
for i in range(5):
    p.apply_async(long_time_task, args=(i, ))
print "Waiting for subprocesses"
p.close()
p.join()
print "All done"

# communication
# Queue and Pipes

from multiprocessing import Process, Queue
import os, time, random

def write(q):
    for value in ["A", "B", "C"]:
        print "Putting %s to queue" % value
        q.put(value)
        time.sleep(random.random())

def read(q):
    while True:
        value = q.get(True)
        print "Getting %s from queue" % value

q = Queue()
pw = Process(target=write, args=(q, ))
pr = Process(target=read, args=(q, ))
pw.start()
pr.start()
pw.join()
pr.terminate()

# pr terminates because it's dead loop

# threading

import time, threading

def loop():
    print "thread %s is running" % threading.current_thread().name
    n = 0
    while n < 5:
        n = n + 1
        print "thread %s >>> %s" % (threading.current_thread().name, n)
        time.sleep(1)
    print "thread %s ended" % threading.current_thread().name

print "thread %s is running" % threading.current_thread().name
t = threading.Thread(target=loop, name="LoopThread")
t.start()
t.join()
print "thread %s ended" % threading.current_thread().name

# thread lock
# unlike multiprocessing where each process has a single copy of the vars
# multithreading shares the same vars
# only the thread that acquires the lock has the privilege to change the var
# and there is only one lock
# must release the lock after you finish
# otherwise deadlock

balance = 0
lock = threading.Lock()

def run_thread(n):
    for i in range(1000):
        lock.acquire()
        try:
            change_it(n)
        finally:
            lock.release()
            
# GIL: Global Interpreter Lock
# it's expected that when we run CPU core number threads, we would occupies all
# the CPU resources. However, Python has GIL and before each thread is run, it 
# must first acquire the GIL and every 100 lines of byte code, GIL is relased.
# Thus multithreading actually is not that multi.
# But GIL has some historical design issues. Normally, folks solve it by 
# multiprocessing, each of which has its own GIL.

# local vars in thread
# we expect threads to have local vars so that we don't need to care about 
# locking global vars; but local vars have problems when in communication
# when use a global dict that maps the thread name to its vars, the code looks
# ugly
# ThreadLocal
# the thread local could be viewed as a dict

import threading

local_school = threading.local()

def process_student():
    print "Hello %s in %s" % (local_school.student, 
                              threading.current_thread().name)

def process_thread(name):
    local_school.student = name
    process_student()

t1 = threading.Thread(target=process_thread, args=("Alic", ), name="Thread_A")
t2 = threading.Thread(target=process_thread, args=("Bob", ), name="Thread_B")
t1.start()
t2.start()
t1.join()
t2.join()
 
# distributed process
# use multiprocessing's managers submodule

# taskmanager.py
import random, time, Queue
from multiprocessing.managers import BaseManager  

task_queue = Queue.Queue()
result_queue = Queue.Queue()

class QueueManager(BaseManager):
    pass

QueueManager.register("get_task_queue", callable=lambda: task_queue)
QueueManager.register("get_result_queue", callable=lambda: result_queue)
# network address, port and key
manager = QueueManager(address=("", 5000), authkey="abc")
manager.start()

task = manager.get_task_queue()
result = manager.get_result.queue()

for i in range(10):
    n = random.randint(0, 10000)
    print "Put task %d" % n
    task.put(n)

print "Getting results"
for i in range(10):
    r = result.get(timeout=10)
    print "Result: %s" % r
manager.shutdown()

# taskworker.py
import time, sys, Queue
from multiprocesssing.managers import BaseManager

class QueueManager(BaseManager):
    pass

QueueManager.register("get_task_queue")
QueueManager.register("get_result_queue")

server_addr = "127.0.0.1"
print "Connecting"

m = QueueManager(address=(server_addr, 5000), authkey="abc")
m.connect()

task = m.get_task_queue()
result = m.get_result.queue()

for i in range(10):
    try:
        n = task.get(timeout=1)
        print "Run task %d * %d" % (n, n)
        r = "%d * %d = %d" % (n, n, n * n)
        time.sleep(1)
        result.put(r)
    except Queue.Empty:
        print "Task queue empty"
print "Exit"

# regular expression
# \d: a digit
# \w: a letter\digit
# \s: a white space
# \_: a underline
# .: any char
# *: a sequence of zero or more chars
# +: a sequence of at least one char
# {n}: a sequence of n chars
# {n, m}: a sequence of n to m chars
# escape: \
# []: range
#     [0-9a-zA-Z\_] one char in this range
# |: or 
#    (P|p)ython
# ^: start with
# $: end with

# re
# encouraged to use r"..."
import re
s = r"ABC\-001"
print re.match(r"^\d{3}\-\d{3, 8}$", "010-12345")
print re.split(r"\s+", "a b     c")
print re.split(r"[\s\,]+", "a,b, c  d")

# grouping, use ()
m = re.match(r"^(\d{3})-(\d{3, 8})$", "010-12345")
# group(0) is the original string
print m.group(0)
print m.group(1)
print m.group(2)

# compile
re_telephone = re.compile(r"^(\d{3})-(\d{3, 8})$")
re_telephone.match("010-12345").groups()

# collections
# namedtuple, deque, defaultdict, OrderedDict, Counter

# itertools
# count, cycle, repeat, groupby, imap, ifilter
# all returns are iterable; could only be accessed by for-loop
# lazy computation with i
# only lazy computation could deal with infinite generator

import itertools
r = itertools.imap(lambda x: x * x, [1, 2, 3])
print r
for x2 in r:
    print x2
    
# GUI
# Tkinter

from Tkinter import *

class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()
    
    def createWidgets(self):
        self.helloLabel = Label(self, text="Hello, World!")
        self.helloLabel.pack()
        self.quitButton = Button(self, text="Quit", command=self.quit)
        self.quitButton.pack()

app = Application()
app.master.title("Hello, World!")
app.mainloop()

# coroutine
# works as if multithreading
# not sequential
# every routine could be interrupted
# but done by the CPU implicitly
# much more efficient than threading cuz no overhead for thread switch
# also no need to care about lock
# cuz all coroutines belong to a single thread
# module gevent
