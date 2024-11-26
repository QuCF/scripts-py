// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.

#nullable enable

using System;
using System.Threading.Tasks;
using System.IO;
using System.Text;
using System.Diagnostics;

using Microsoft.Quantum.Simulation.Core;
using Microsoft.Quantum.Simulation.Simulators;
using Microsoft.Quantum.Research.QuantumSignalProcessing;

namespace Microsoft.Quantum.Research.Samples
{
    public class Host
    {
        static async Task Main(string[] args)
        {
            // printfn "alphaSqrt/2 = %O" alphaSqrt.Half
            Console.WriteLine("Start calculating...");
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            // var tau = 5.31;
            // var eps = 1.0e-20;

            // var tau = 10;
            // var eps = 1.0e-10;

            // var tau = 100;
            // var eps = 1.0e-10;

            // var tau = 200;
            // var eps = 1.0e-4;

            // Read tau and eps:
            var tau = System.Convert.ToDouble(args[0]);
            var eps = System.Convert.ToDouble(args[1]);
            var write_to = args[2];
            var fname_loc = args[3];

            // --- Quantum singal processing ---
            var qspResult = QSP.JacobiAngerExpansion(eps, tau);
            var anglesRes = new QArray<double>( QSP.ConvertToAnglesForParity01(qspResult) );

            var Na = anglesRes.Length;

            // --- output the angles on Console ---
            Console.WriteLine("\nResulting angles: ");
            foreach(var ii in anglesRes)
            {
                Console.Write("{0}   ", ii);
            }

            // ---- Write down the angles into a text file ----
            // var fileName = "/home/ivan/Documents/Postdoc-PPPL/progs/python/QCSyn/Naah-angles.dat";
            var fileName = write_to + "/" + fname_loc;
            FileStream fs = new FileStream(fileName, FileMode.OpenOrCreate, FileAccess.Write);
            Console.WriteLine("\n\nFile created");
            fs.Close();

            StreamWriter sw = new StreamWriter(fileName);
            sw.WriteLine("{0}", tau);
            sw.WriteLine("{0}", eps);
            sw.WriteLine("{0}", Na);
            foreach(var ii in anglesRes)
            {
                sw.WriteLine("{0}", ii);
            }
            sw.Close();

            // // --- Check the angles by performing Hamiltonian simulations ---
            // Console.WriteLine("\n\nCheck results...");
            // using var qsim = new QuantumSimulator();
            // await SampleHamiltonianEvolutionByQSP.Run(qsim, tau, anglesRes);
            // Console.WriteLine("JacobiAnger QSP Done.");

            stopwatch.Stop();
            var t_calc = stopwatch.ElapsedMilliseconds/1000;
            Console.WriteLine("\nElapsed Time is {0} s", t_calc);
        }

    }
}
