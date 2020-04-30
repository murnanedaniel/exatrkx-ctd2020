# CTD2020 ExatrkX

![](Docs/pipeline.png)

```
            Build
     +-----------------+


   +---------------------+
   |   Metric Learning   |
   +----------+----------+
              |
              |
              |
              v

       buildDoublets.py

              +
              |
              |
              |
              v
   +----------+---------+
   |   GraphLearning    |
   +----------+---------+
              |
              |
              |
              v

        train.py doubletAGNN.yaml

              +
              |
              v


       buildTriplets.py

              +
              |
              v

        train.py tripletAGNN.yaml

                  +
                  |
         +--------+-------+
         |                |
         |                |
         |                |
         |                |
         v                v

+------------+       +-----------+
|  Seeding   |       | Labelling |
+------------+       +-----------+

      +                    +
      |                    |
      |                    |
      v                    v

   seed.py              label.py
     config.yaml           config.yaml

```
