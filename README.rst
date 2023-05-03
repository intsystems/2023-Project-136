|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Стохастический метод Ньютона с различным семплингом
    Stochastic Newton with Arbitrary Sampling
    :Тип научной работы: M1P
    :Автор: Денис Александрович Швейкин
    :Научный руководитель: Исламов Рустем Ильфакович
    
Аннотация
========
Задача минимизации среднего от большого числа гладких сильно выпуклых функций встречается в машинном обучении повсеместно. Стохастические методы первого порядка, такие как стохастический градиентный спуск (SGD), для этой задачи хорошо изучены. В свою очередь, методы второго порядка, такие как метод Ньютона имеют определенные преимущества, поскольку могут адаптироваться к кривизне задачи. Также они известны своей быстрой сходимостью. Однако стохастические варианты методов типа Ньютон изучены не так хорошо, как методы типа SGD, и имеют ограничения на размеры батчей. Ранее был предложен метод, который не требует таких ограничений. Наша работа исследует этот метод с различными стратегиями семплинга, которые ведут практическим улучшениям.

The problem of minimizing the average of a large number of sufficiently smooth and strongly convex functions is ubiquitous in machine learning. Stochastic first-order methods for this problem of Stochastic Gradient Descent type are well studied. In turn, second-order methods, such as Newton, have certain advances since they can adapt to the curvature of the problem. They are also known for their fast convergence rates. But stochastic variants of Newton-type methods are not studied as good as SGD-type ones and have limitations on the batch size. Previously, a method was proposed which requires no limitations on batch sizes. Our work explores this method with different sampling strategies that lead to practical improvements.
