using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SISE_TWO
{
    class Sigmoid
    {
        public static double Derivative(double x)
        {
            return (1 - x) * x;
        }

        public static double Func(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }
    }
}
