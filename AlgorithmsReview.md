# Алгоритмы оптимизации с реализацией

## Неускоренные методы первого порядка


- ### метод двойственных усреднений ([=](https://arxiv.org/pdf/1604.08183.pdf) зеркальный спуск)

  [Зеркальный спуск на python с fmin.xyz](https://fmin.xyz/docs/methods/fom/Mirror_descent/)

  [Зеркальный спуск на c++ с github](https://github.com/ChandlerLutz/MirrorDescSynth/blob/master/src/mirror_desc.cpp "автор Chandler Lutz, phd по экономике")  

## Ускоренные методы первого порядка

- ### метод Нестерова 

  [Метод Нестерова на c++ с github](https://github.com/hughperkins/DeepCL/blob/master/src/trainers/Nesterov.cpp)

  [Метод Нестерова на python с сайта Катруцы](https://nbviewer.jupyter.org/github/amkatrutsa/MIPT-Opt/blob/master/ODE4NesterovAcc/ODE4NesterovAcc.ipynb "и его применения для решения дифуров")

  

- ### прочее

  [Метод сопряжённых градиентов, тяжёлого шарика и ускоренный метод Нестерова – python, сайт Катруцы](https://github.com/amkatrutsa/MIPT-Opt/blob/master/Spring2020/acc_grad.ipynb)

## Квазиньютоновские методы

  - [Квазиньютоновские (и ньютоновский) методы, python, сайт Катруцы](https://github.com/amkatrutsa/MIPT-Opt/blob/master/Spring2020/newton_quasi.ipynb)

  - [Сравнение квазиньютоновских методов, python, fmin.xyz](https://nbviewer.jupyter.org/github/fabianp/pytron/blob/master/doc/benchmark_logistic.ipynb)

  - [BFGS, L-FGS и Ньютоновский метод - сравнение и реализация на cpp](http://dlib.net/optimization_ex.cpp.html)

## Другие методы второго порядка:

- ### метод сопряженных градиентов

  [c++ библиотека метода сопряженных градиентов](https://people.sc.fsu.edu/~jburkardt/cpp_src/cg/cg.html)

  [Метод сопряженных градиентов, python, github](https://gist.github.com/sfujiwara/b135e0981d703986b6c2)

- ### метод Ньютона-Гаусса

  [реализация на c++, github](https://github.com/omyllymaki/solvers/tree/master/src/numerical/gauss-newton)

  [реализация на python, github](https://github.com/basil-conto/gauss-newton)