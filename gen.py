
"""
notes
 - jacobian
   to jacobian just puts extra coordinate z=1
   to recover affine from jacobian, need reciprocal_fp which is a square and mul loop (like double and add) over exponent bits (which is hard-coded, related to field prime). fp2 uses one of these too.
 - montgomery form
   - I think that you can just use, but must reduce twice to recover
   - reduce by doing mulmodmont with other input=1
"""


#######
# utils

def gen_copy(dst_offset,src_offset,len_):
  if len_<32:
    print("ERROR gen_copy() len_ is ",len_)
    return
  while len_>32:
    len_-=32
    print(hex(src_offset))
    print("mload")
    print(hex(dst_offset))
    print("mstore")
    src_offset+=32
    dst_offset+=32
  print(hex(src_offset-(32-len_)))
  print("mload")
  print(hex(dst_offset-(32-len_)))
  print("mstore")

def gen_isNonzero(offset,len_):
  # leaves stack item 0 if zero or >0 if nonzero
  # len_ must be >=33 bytes
  if len_<32:
    print("ERROR gen_isZero() len_ is ",len_)
    return
  print(hex(offset))
  print("mload")
  print("iszero 0x1 sub")
  buffer_+=32
  len_-=32
  while len_>32:
    print(hex(offset))
    print("mload")
    print("iszero 0x1 sub")
    print("add")
    buffer_+=32
    len_-=32
  # final check
  if len_>0:
    print(hex(buffer_-(32-len_)))
    print("mload")
    print("iszero 0x1 sub")
    print("add")


def gen_bit_iter():
  """
  push 0xblah	// bits we are iterating over
  jumpdest	// bit iter func, top of stack must be the bits we are iterating over
  push 0x1	// mask to isolate rightmost bit
  push 0x0	// loop variable, what we shift by
  jumpdest	// start bit iter loop
  dup1 dup1	// stack: [... <val to shift> <numbits to shift by>]
  shr
  dup2 and	// apply mask
  0x1 sub	// top of stack is 1 if igore, zero if must do
  // if not zero, addmod384, else jump forward
  0x
  0x<location of start bit iter loop> mul
  add jump
  // double
  jumpdest	// after 
  
  
  jump <start of loop or forward by one>
  jumpdest
  """
  pass

bls12_384_prime = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

# offsets for zero, input/output, and local buffers
zero = 0
mod = 100
buffer_f2mul = 1000
buffer_E2add = 2000
buffer_E2double = 3000


def gen_evm384_offsets(a,b,c,d):
  print("0x"+hex(a)[2:].zfill(8)+hex(b)[2:].zfill(8)+hex(c)[2:].zfill(8)+hex(d)[2:].zfill(8), end=' ')


addmod384_count=0
submod384_count=0
mulmodmont384_count=0
f2add_count=0
f2sub_count=0
f2mul_count=0
f6add_count=0
f6add_count=0
f6mul_count=0

def gen_fadd(f,out,x,y,mod):
  if f=="f6":
    gen_f6add(out,x,y,mod)
  if f=="f2":
    gen_f2add(out,x,y,mod)
  if f=="f1":
    gen_f1add(out,x,y,mod)

def gen_fsub(f,out,x,y,mod):
  if f=="f6":
    gen_f6sub(out,x,y,mod)
  if f=="f2":
    gen_f2sub(out,x,y,mod)
  if f=="f1":
    gen_f1sub(out,x,y,mod)

def gen_fmul(f,out,x,y,mod):
  if f=="f6":
    gen_f6mul(out,x,y,mod)
  if f=="f2":
    gen_f2mul(out,x,y,mod)
  if f=="f1":
    gen_f1mul(out,x,y,mod)


#################################
## Field operations add, sub, mul

# f1

def gen_f1add(out,x,y,mod):
  global addmod384_count
  gen_evm384_offsets(out,x,y,mod); print("addmod384"); addmod384_count+=1

def gen_f1sub(out,x,y,mod):
  global submod384_count
  gen_evm384_offsets(out,x,y,mod); print("submod384"); submod384_count+=1

def gen_f1mul(out,x,y,mod):
  global mulmodmont384_count
  gen_evm384_offsets(out,x,y,mod); print("mulmodmont384"); mulmodmont384_count+=1
  
def gen_f1reciprocal(out_offset,in_offset,mod):
  pass
  #for bit in bin(bls12_384_prime-2)[2]:
  #  if bit:
  """
  limbs=[0xb9feffffffffaaa9,0x1eabfffeb153ffff,0x6730d2a0f6b0f624,0x64774b84f38512bf,0x4b1ba7b6434bacd7,0x1a0111ea397fe69a]
  for e in limbs:
    print()
  """




# f2

def gen_f2add(out,x,y,mod):
  global f2add_count
  f2add_count+=1
  print("// f2 add")
  x0 = x
  x1 = x+48
  y0 = y
  y1 = y+48
  out0 = out
  out1 = out+48
  gen_f1add(out0,x0,y0,mod)
  gen_f1add(out1,x1,y1,mod)

def gen_f2sub(out,x,y,mod):
  global f2sub_count
  f2sub_count+=1
  print("// f2 sub")
  x0 = x
  x1 = x+48
  y0 = y
  y1 = y+48
  out0 = out
  out1 = out+48
  gen_f1sub(out0,x0,y0,mod)
  gen_f1sub(out1,x1,y1,mod)

def gen_f2mul(out,x,y,mod):
  global f2mul_count
  f2mul_count+=1
  print("// f2 mul")
  # get offsets
  x0 = x
  x1 = x+48
  y0 = y
  y1 = y+48
  out0 = out
  out1 = out+48
  tmp1 = buffer_f2mul
  tmp2 = tmp1+48
  tmp3 = tmp2+48
  # tmp1 = x0*y0
  gen_f1mul(tmp1,x0,y0,mod)
  # tmp2 = x1*y1
  gen_f1mul(tmp2,x1,y1,mod)
  # tmp3 = zero-tmp2
  gen_f1sub(tmp3,zero,tmp2,mod)
  # out0 = tmp1+tmp3
  gen_f1add(tmp3,zero,y1,mod)
  # tmp1 = tmp1+tmp2
  gen_f1add(tmp1,tmp1,tmp2,mod)
  # tmp2 = x0+x1
  gen_f1add(tmp2,x0,x1,mod)
  # tmp3 = y0+y1
  gen_f1add(tmp3,y0,y1,mod)
  # tmp2 = tmp2*tmp3
  gen_f1mul(tmp2,tmp2,tmp3,mod)
  # out1 = tmp2-tmp1
  gen_f1sub(out1,tmp2,tmp1,mod)

# f6

def gen_f6add(out,x,y,mod):
  global f6add_count
  f6add_count+=1
  print("// f6 add")
  x0 = x
  x1 = x0+48
  x2 = x1+48
  x3 = x2+48
  x4 = x3+48
  x5 = x4+48
  y0 = y
  y1 = y0+48
  y2 = y1+48
  y3 = y2+48
  y4 = y3+48
  y5 = y4+48
  out0 = out
  out1 = out0+48
  out2 = out1+48
  out3 = out2+48
  out4 = out3+48
  out5 = out4+48
  gen_f1add(out0,x0,y0,mod)
  gen_f1add(out1,x1,y1,mod)
  gen_f1add(out2,x2,y2,mod)
  gen_f1add(out3,x3,y3,mod)
  gen_f1add(out4,x4,y4,mod)
  gen_f1add(out5,x5,y5,mod)

def gen_f6sub(out,x,y,mod):
  global f6sub_count
  f6add_count+=1
  print("// f6 add")
  x0 = x
  x1 = x0+48
  x2 = x1+48
  x3 = x2+48
  x4 = x3+48
  x5 = x4+48
  y0 = y
  y1 = y0+48
  y2 = y1+48
  y3 = y2+48
  y4 = y3+48
  y5 = y4+48
  out0 = out
  out1 = out0+48
  out2 = out1+48
  out3 = out2+48
  out4 = out3+48
  out5 = out4+48
  gen_f1sub(out0,x0,y0,mod)
  gen_f1sub(out1,x1,y1,mod)
  gen_f1sub(out2,x2,y2,mod)
  gen_f1sub(out3,x3,y3,mod)
  gen_f1sub(out4,x4,y4,mod)
  gen_f1sub(out5,x5,y5,mod)

def gen_f6sub(out,x,y,mod):
  global f6sub_count
  f6add_count+=1
  print("// f6 add")
  x0 = x
  x1 = x0+48
  x2 = x1+48
  x3 = x2+48
  x4 = x3+48
  x5 = x4+48
  y0 = y
  y1 = y0+48
  y2 = y1+48
  y3 = y2+48
  y4 = y3+48
  y5 = y4+48
  out0 = out
  out1 = out0+48
  out2 = out1+48
  out3 = out2+48
  out4 = out3+48
  out5 = out4+48
  gen_f1sub(out0,x0,y0,mod)
  gen_f1sub(out1,x1,y1,mod)
  gen_f1sub(out2,x2,y2,mod)
  gen_f1sub(out3,x3,y3,mod)
  gen_f1sub(out4,x4,y4,mod)
  gen_f1sub(out5,x5,y5,mod)

def gen_f6mul(out,x,y,mod):
  global f6mul_count
  f6mul_count+=1
  print("// f6 add")
  x0 = x
  x1 = x0+48
  x2 = x1+48
  x3 = x2+48
  x4 = x3+48
  x5 = x4+48
  y0 = y
  y1 = y0+48
  y2 = y1+48
  y3 = y2+48
  y4 = y3+48
  y5 = y4+48
  out0 = out
  out1 = out0+48
  out2 = out1+48
  out3 = out2+48
  out4 = out3+48
  out5 = out4+48




###############################
# Curve operations: add, double

def gen_Eadd__madd_2001_b(f,XYZout,XYZ1,XYZ2,buffer_):
  print("/////////")
  print("// Eadd https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2001-b")
  # inputs/ouput
  X1=XYZ1
  Y1=X1+int(f[1])*48
  Z1=Y1+int(f[1])*48
  X2=XYZ2
  Y2=X2+int(f[1])*48
  Z2=Y2+int(f[1])*48
  X3=XYZout
  Y3=X3+int(f[1])*48
  Z3=Y3+int(f[1])*48
  """
  ZZ1 = Z1^2
  ZZZ1 = Z1*ZZ1
  ZZ2 = Z2^2
  ZZZ2 = Z2*ZZ2
  A = X1*ZZ2
  B = X2*ZZ1-A
  c = Y1*ZZZ2
  d = Y2*ZZZ1-c
  e = B^2
  f = B*e
  g = A*e
  h = Z1*Z2
  f2g = 2*g+f
  X3 = d^2-f2g
  Z3 = B*h
  gx = g-X3
  Y3 = d*gx-c*f
  """
  # temp vars
  ZZ1 = buffer_
  ZZZ1 = ZZ1+int(f[1])*48
  ZZ2 = ZZZ1+int(f[1])*48
  ZZZ2 = ZZ2+int(f[1])*48
  A = ZZZ2+int(f[1])*48
  B = A+int(f[1])*48
  c = B+int(f[1])*48
  d = c+int(f[1])*48
  e = d+int(f[1])*48
  f_ = e+int(f[1])*48
  g = f_+int(f[1])*48
  h = g+int(f[1])*48
  f2g = h+int(f[1])*48
  gx = f2g+int(f[1])*48

  print("ZZ1 = Z1^2")
  gen_fmul(f,ZZ1,Z1,Z1,mod)
  print("ZZZ1 = Z1*ZZ1")
  gen_fmul(f,ZZZ1,Z1,ZZ1,mod)
  print("ZZ2 = Z2^2")
  gen_fmul(f,ZZ2,Z2,Z2,mod)
  print("ZZZ2 = Z2*ZZ2")
  gen_fmul(f,ZZZ2,Z2,ZZ2,mod)
  print("A = X1*ZZ2")
  gen_fmul(f,A,X1,ZZ2,mod)
  print("B = X2*ZZ1-A")
  gen_fmul(f,B,X2,ZZ1,mod)
  gen_fsub(f,B,B,A,mod)
  print("c = Y1*ZZZ2")
  gen_fmul(f,c,Y1,ZZZ2,mod)
  print("d = Y2*ZZZ1-c")
  gen_fmul(f,d,Y2,ZZZ1,mod)
  gen_fsub(f,d,d,c,mod)
  print("e = B^2")
  gen_fmul(f,e,B,B,mod)
  print("f = B*e")
  gen_fmul(f,f_,B,e,mod)
  print("g = A*e")
  gen_fmul(f,g,A,e,mod)
  print("h = Z1*Z2")
  gen_fmul(f,h,Z1,Z2,mod)
  print("f2g = 2*g+f")
  gen_fadd(f,f2g,g,g,mod)
  gen_fadd(f,f2g,f2g,f_,mod)
  print("X3 = d^2-f2g")
  gen_fmul(f,X3,d,d,mod)
  gen_fsub(f,X3,X3,f2g,mod)
  print("Z3 = B*h")
  gen_fmul(f,Z3,B,h,mod)
  print("gx = g-X3")
  gen_fsub(f,gx,g,X3,mod)
  print("Y3 = d*gx-c*f")
  gen_fmul(f,Y3,d,g,mod)
  gen_fmul(f,c,c,f_,mod)	# clobber c
  gen_fsub(f,Y3,Y3,c,mod)

  print("// E add")
  print("/////////")

def gen_Eadd__madd_2007_bl(f,XYZout,XYZ1,XYZ2,buffer_):
  print("/////////")
  print("// Eadd https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl")

  # inputs/ouput
  X1=XYZ1
  Y1=X1+int(f[1])*48
  Z1=Y1+int(f[1])*48
  X2=XYZ2
  Y2=X2+int(f[1])*48
  Z2=Y2+int(f[1])*48
  X3=XYZout
  Y3=X3+int(f[1])*48
  Z3=Y3+int(f[1])*48

  # temp vars
  Z1Z1 = buffer_
  U2 = Z1Z1+int(f[1])*48
  S2 = U2+int(f[1])*48
  H = S2+int(f[1])*48
  HH = H+int(f[1])*48
  I = HH+int(f[1])*48
  J = I+int(f[1])*48
  r = J+int(f[1])*48
  V = r+int(f[1])*48

  # Z1Z1 = Z1^2
  print("// Z1Z1 = Z1^2")
  gen_fmul(f,Z1Z1,Z1,Z1,mod)
  # U2 = X2*Z1Z1
  print("// U2 = X2*Z1Z1")
  gen_fmul(f,U2,X2,Z1Z1,mod)
  # S2 = Y2*Z1*Z1Z1
  print("// S2 = Y2*Z1*Z1Z1")
  gen_fmul(f,S2,Z1,Z1Z1,mod)
  gen_fmul(f,S2,S2,Y2,mod)
  # H = U2-X1
  print("// H = U2-X1")
  gen_fsub(f,H,U2,X1,mod)
  # HH = H^2
  print("// HH = H^2")
  gen_fmul(f,HH,H,H,mod)
  # I = 4*HH
  print("// I = 4*HH")
  gen_fadd(f,I,HH,HH,mod)
  gen_fadd(f,I,I,I,mod)
  # J = H*I
  print("// J = H*I")
  gen_fmul(f,J,H,I,mod)
  # r = 2*(S2-Y1)
  print("// r = 2*(S2-Y1)")
  gen_fsub(f,r,S2,Y1,mod)
  gen_fadd(f,r,r,r,mod)
  # V = X1*I
  print("// V = X1*I")
  gen_fmul(f,V,X1,I,mod)
  # X3 = r^2-J-2*V
  print("// X3 = r^2-J-2*V")
  gen_fmul(f,X3,r,r,mod)
  gen_fsub(f,X3,X3,J,mod)
  gen_fsub(f,X3,X3,V,mod)
  gen_fsub(f,X3,X3,V,mod)
  # Y3 = r*(V-X3)-2*Y1*J
  print("// Y3 = r*(V-X3)-2*Y1*J")
  gen_fsub(f,Y3,V,X3,mod)
  gen_fmul(f,Y3,r,Y3,mod)
  gen_fmul(f,J,Y1,J,mod)	# overwriting J
  gen_fsub(f,Y3,Y3,J,mod)
  gen_fsub(f,Y3,Y3,J,mod)
  # Z3 = (Z1+H)^2-Z1Z1-HH
  print("// Z3 = (Z1+H)^2-Z1Z1-HH")
  gen_fadd(f,Z3,Z1,H,mod)
  gen_fmul(f,Z3,Z3,Z3,mod)
  gen_fsub(f,Z3,Z3,Z1Z1,mod)
  gen_fsub(f,Z3,Z3,HH,mod)
  
  print("// E add")
  print("/////////")




def gen_Edouble__dbl_2009_alnr(f,XYZout,XYZ,buffer_):
  print("///////////")
  print("// Edouble https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-alnr")

  # inputs/ouput
  X1=XYZ
  Y1=X1+int(f[1])*48
  Z1=Y1+int(f[1])*48
  X3=XYZout
  Y3=X3+int(f[1])*48
  Z3=Y3+int(f[1])*48

  """
  A = X1^2
  B = Y1^2
  ZZ = Z1^2
  C = B^2
  D = 2*((X1+B)^2-A-C)
  E = 3*A
  F = E^2
  X3 = F-2*D
  Y3 = E*(D-X3)-8*C
  Z3 = (Y1+Z1)^2-B-ZZ
  """
  A = buffer_
  B = A+int(f[1])*48
  ZZ = B+int(f[1])*48 
  C = ZZ+int(f[1])*48
  D = C+int(f[1])*48
  E = D+int(f[1])*48
  F = E+int(f[1])*48

  print("// A = X1^2")
  gen_fmul(f,A,X1,X1,mod)
  print("// B = Y1^2")
  gen_fmul(f,B,Y1,Y1,mod)
  print("// ZZ = Z1^2")
  gen_fmul(f,ZZ,Z1,Z1,mod)
  print("// C = B^2")
  gen_fmul(f,C,B,B,mod)
  print("// D = 2*((X1+B)^2-A-C)")
  gen_fadd(f,D,X1,B,mod)
  gen_fmul(f,D,D,D,mod)
  gen_fsub(f,D,D,A,mod)
  gen_fsub(f,D,D,C,mod)
  gen_fadd(f,D,D,D,mod)
  print("// E = 3*A")
  gen_fadd(f,E,A,A,mod)
  gen_fadd(f,E,E,A,mod)
  print("// F = E^2")
  gen_fmul(f,F,E,E,mod)
  print("// X3 = F-2*D")
  gen_fadd(f,X3,D,D,mod)
  gen_fsub(f,X3,F,X3,mod)
  print("// Y3 = E*(D-X3)-8*C")
  gen_fsub(f,Y3,D,X3,mod)
  gen_fmul(f,Y3,E,Y3,mod)
  gen_fadd(f,C,C,C,mod)		# overwriting C
  gen_fadd(f,C,C,C,mod)
  gen_fadd(f,C,C,C,mod)
  gen_fsub(f,Y3,Y3,C,mod)
  print("// Z3 = (Y1+Z1)^2-B-ZZ")
  gen_fadd(f,Z3,Y1,Z1,mod)
  gen_fmul(f,Z3,Z3,Z3,mod)
  gen_fsub(f,Z3,Z3,B,mod)
  gen_fsub(f,Z3,Z3,ZZ,mod)
  print("// E double")
  print("////////////")


def gen_Edouble__dbl_2009_l(f,XYZout,XYZ,buffer_):
  print("///////////")
  print("// Edouble https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l")

  # inputs/ouput
  X1=XYZ
  Y1=X1+int(f[1])*48
  Z1=Y1+int(f[1])*48
  X3=XYZout
  Y3=X3+int(f[1])*48
  Z3=Y3+int(f[1])*48

  """
  A = X1^2
  B = Y1^2
  C = B^2
  D = 2*((X1+B)^2-A-C)
  E = 3*A
  F = E^2
  X3 = F-2*D
  Y3 = E*(D-X3)-8*C
  Z3 = 2*Y1*Z1
  """
  A = buffer_
  B = A+int(f[1])*48
  C = B+int(f[1])*48
  D = C+int(f[1])*48
  E = D+int(f[1])*48
  F = E+int(f[1])*48

  print("// A = X1^2")
  gen_fmul(f,A,X1,X1,mod)
  print("// B = Y1^2")
  gen_fmul(f,B,Y1,Y1,mod)
  print("// C = B^2")
  gen_fmul(f,C,B,B,mod)
  print("// D = 2*((X1+B)^2-A-C)")
  gen_fadd(f,D,X1,B,mod)
  gen_fmul(f,D,D,D,mod)
  gen_fsub(f,D,D,A,mod)
  gen_fsub(f,D,D,C,mod)
  gen_fadd(f,D,D,D,mod)
  print("// E = 3*A")
  gen_fadd(f,F,A,A,mod)
  gen_fadd(f,F,F,A,mod)
  print("// F = E^2")
  gen_fmul(f,F,E,E,mod)
  print("// X3 = F-2*D")
  gen_fadd(f,X3,D,D,mod)
  gen_fsub(f,X3,F,D,mod)
  print("// Y3 = E*(D-X3)-8*C")
  gen_fsub(f,Y3,D,X3,mod)
  gen_fmul(f,Y3,E,Y3,mod)
  gen_fadd(f,C,C,C,mod)		# clobber C
  gen_fadd(f,C,C,C,mod)
  gen_fadd(f,C,C,C,mod)
  gen_fsub(f,Y3,Y3,C,mod)
  print("// Z3 = 2*Y1*Z1")
  gen_fmul(f,Z3,Y1,Z1,mod)
  gen_fadd(f,Z3,Z3,Z3,mod)
  print("// E double")
  print("////////////")


if __name__=="__main__":
  print()
  print()
  #gen_Eadd__madd_2007_bl("f2",1,2,3,4)
  print()
  print()
  #gen_Edouble__dbl_2009_alnr("f2",1,2,3)
  print()
  print()
  #gen_Eadd__madd_2007_bl("f1",1,2,3,4)
  print()
  print()
  addmod384_count=0
  submod384_count=0
  mulmodmont384_count=0
  #gen_Edouble__dbl_2009_alnr("f2",1,2,3)
  gen_Eadd__madd_2001_b("f2",1,2,3,4)
  print("addmod384_count ",addmod384_count)
  print("submod384_count ",submod384_count)
  print("mulmodmont384_count ",mulmodmont384_count)
  print()
  print()
  addmod384_count=0
  submod384_count=0
  mulmodmont384_count=0
  #gen_Edouble__dbl_2009_l("f2",1,2,3)
  gen_Eadd__madd_2007_bl("f2",1,2,3,4)
  print("addmod384_count ",addmod384_count)
  print("submod384_count ",submod384_count)
  print("mulmodmont384_count ",mulmodmont384_count)
  #global f2add_count
  #global f2sub_count
  #global f2mul_count
  """
  print("f2add_count ",f2add_count)
  print("f2sub_count ",f2sub_count)
  print("f2mul_count ",f2mul_count)
  print("addmod384_count ",addmod384_count)
  print("submod384_count ",submod384_count)
  print("mulmodmont384_count ",mulmodmont384_count)
  """


