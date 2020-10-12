using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Division
{
    class Program
    {
        static void Main(string[] args)
        {
            int int1, int2, res;

            Console.WriteLine("Welcome to Integer Division!");

            Console.WriteLine("Kindly input your first integer:");
            int1 = Int32.Parse(Console.ReadLine());
            Console.WriteLine("Kindly input your second integer that isn't 0:");
            int2 = Int32.Parse(Console.ReadLine());

            res = int1 / int2;
            Console.WriteLine("Your result is " + res);
        }
    }
}
