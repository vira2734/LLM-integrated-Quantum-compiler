OPENQASM 2.0;
include "qelib1.inc";
gate custom4 a,b,c,d { }
gate custom5 a,b,c,d,e { }
qreg q[5];
h q[0];
cx q[0], q[1];
ccx q[0], q[1], q[2];
custom4 q[0], q[1], q[2], q[3];
custom5 q[0], q[1], q[2], q[3], q[4];
x q[0];
cx q[1], q[0];
