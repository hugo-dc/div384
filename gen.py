
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

def gen_memstore(dst_offset,bytes_):
  idx = 0
  if len(bytes_)<32:
    print("ERROR gen_copy() len_ is ",len_)
    return
  while idx<len(bytes_)-32:
    print("0x"+bytes_[idx:idx+32].hex())
    print(hex(dst_offset))
    print("mstore")
    dst_offset+=32
    idx+=32
  print("0x"+bytes_[-32:].hex())
  print(hex(dst_offset+len(bytes_)-32))
  print("mstore")

def gen_memcopy(dst_offset,src_offset,len_):
  if len_<32:
    print("ERROR gen_memcopy() len_ is ",len_)
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



###########
# Constants

bls12_384_prime = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

# offsets for zero, input/output, and local buffers
buffer_offset = 0
zero = buffer_offset
f1zero = buffer_offset	# 48 bytes
f2zero = buffer_offset	# 96 bytes
f6zero = buffer_offset	# 288 bytes
f12zero = buffer_offset	# 576 bytes
buffer_offset += 576
f12one = buffer_offset	# 576 bytes
buffer_offset += 576
mod = buffer_offset	# 56 bytes, mod||inv
buffer_offset += 56
buffer_miller_loop = buffer_offset	# 1 E2 point, 1 E1 point affine
buffer_offset += 288+96
buffer_line = buffer_offset		# 3 f2 points
buffer_offset += 288
buffer_f2mul = buffer_offset	# 3 f1 points
buffer_offset += 144
buffer_f6mul = buffer_offset	# 6 f2 points
buffer_offset += 576
buffer_f12mul = buffer_offset	# 3 f6 points
buffer_offset += 864
buffer_Eadd = buffer_offset	# 14 or 9 values
buffer_offset += 14*3*96
buffer_Edouble = buffer_offset	# 7 or 6 values
buffer_offset += 7*3*96
buffer_inputs = buffer_offset
buffer_offset += 3*48+3*96
buffer_output = buffer_offset
buffer_offset += 12*48


# for counting number of operations
addmod384_count=0
submod384_count=0
mulmodmont384_count=0
f2add_count=0
f2sub_count=0
f2mul_count=0
f6add_count=0
f6sub_count=0
f6mul_count=0
f12add_count=0
f12sub_count=0
f12mul_count=0


# pack offsets into stack item
def gen_evm384_offsets(a,b,c,d):
  print("0x"+hex(a)[2:].zfill(8)+hex(b)[2:].zfill(8)+hex(c)[2:].zfill(8)+hex(d)[2:].zfill(8), end=' ')



#################################
## Field operations add, sub, mul

# general ops when field can change

def gen_fadd(f,out,x,y,mod):
  if f=="f12":
    gen_f12add(out,x,y,mod)
  if f=="f6":
    gen_f6add(out,x,y,mod)
  if f=="f2":
    gen_f2add(out,x,y,mod)
  if f=="f1":
    gen_f1add(out,x,y,mod)

def gen_fsub(f,out,x,y,mod):
  if f=="f12":
    gen_f12sub(out,x,y,mod)
  if f=="f6":
    gen_f6sub(out,x,y,mod)
  if f=="f2":
    gen_f2sub(out,x,y,mod)
  if f=="f1":
    gen_f1sub(out,x,y,mod)

def gen_fmul(f,out,x,y,mod):
  if f=="f12":
    gen_f12mul(out,x,y,mod)
  if f=="f6":
    gen_f6mul(out,x,y,mod)
  if f=="f2":
    gen_f2mul(out,x,y,mod)
  if f=="f1":
    gen_f1mul(out,x,y,mod)


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
  
def gen_f1neg(out,x,mod):
  global submod384_count
  gen_evm384_offsets(out,f1zero,x,mod); print("submod384"); submod384_count+=1

def gen_f1reciprocal(out_offset,in_offset,mod):
  pass
  #for bit in bin(bls12_384_prime-2)[2]:
  #  if bit:




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
  # temporary values
  tmp1 = buffer_f2mul
  tmp2 = tmp1+48
  tmp3 = tmp2+48
  # tmp1 = x0*y0
  gen_f1mul(tmp1,x0,y0,mod)
  # tmp2 = x1*y1
  gen_f1mul(tmp2,x1,y1,mod)
  # tmp3 = zero-tmp2
  gen_f1sub(tmp3,f2zero,tmp2,mod)
  # out0 = tmp1+tmp3
  gen_f1add(tmp3,f2zero,y1,mod)
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

def gen_f2sqr(out,x,mod):
  gen_f2mul(out,x,x,mod)	# TODO: optimize

def gen_f2neg(out,in_,mod):
  gen_f2sub(out,zero,in_,mod)

def gen_mul_by_u_plus_1_fp2(out,x,mod):
  gen_f1sub(out, x, x+48, mod)
  gen_f1add(out+48, x, x+48, mod)


# f6

def gen_f6add(out,x,y,mod):
  global f6add_count
  f6add_count+=1
  print("// f6 add")
  x0 = x
  x1 = x0+96
  x2 = x1+192
  y0 = y
  y1 = y0+96
  y2 = y1+192
  out0 = out
  out1 = out0+96
  out2 = out1+192
  gen_f2add(out0,x0,y0,mod)
  gen_f2add(out1,x1,y1,mod)
  gen_f2add(out2,x2,y2,mod)

def gen_f6sub(out,x,y,mod):
  global f6sub_count
  f6sub_count+=1
  print("// f6 sub")
  x0 = x
  x1 = x0+96
  x2 = x1+192
  y0 = y
  y1 = y0+96
  y2 = y1+192
  out0 = out
  out1 = out0+96
  out2 = out1+192
  gen_f2sub(out0,x0,y0,mod)
  gen_f2sub(out1,x1,y1,mod)
  gen_f2sub(out2,x2,y2,mod)

def gen_f6neg(out,x,mod):
  gen_f6sub(out,f6zero,x,mod)

def gen_f6mul(out,x,y,mod):
  global f6mul_count
  f6mul_count+=1
  print("// f6 add")
  x0 = x
  x1 = x0+96
  x2 = x1+96
  y0 = y
  y1 = y0+96
  y2 = y1+96
  out0 = out
  out1 = out0+96
  out2 = out1+96
  # temporary variables
  t0 = buffer_f6mul
  t1 = t0+96
  t2 = t1+96
  t3 = t2+96
  t4 = t3+96
  t5 = t4+96
  # algorithm
  gen_f2mul(t0,x0,y0,mod)
  gen_f2mul(t1,x1,y1,mod)
  gen_f2mul(t2,x2,y2,mod)
  # out0
  gen_f2add(t4,x1,x2,mod)
  gen_f2add(t5,y1,y2,mod)
  gen_f2mul(t3,t4,t5,mod)
  gen_f2sub(t3,t3,t1,mod)
  gen_f2sub(t3,t3,t2,mod)
  gen_mul_by_u_plus_1_fp2(t3,t3,mod)
  gen_f2add(out0,t3,t0,mod)
  # out1
  gen_f2add(t4,x0,x1,mod)
  gen_f2add(t5,y0,y1,mod)
  gen_f2mul(out1,t4,t5,mod)
  gen_f2sub(out1,out1,t0,mod)
  gen_f2sub(out1,out1,t1,mod)
  gen_mul_by_u_plus_1_fp2(t4,t2,mod)
  gen_f2add(out1,out1,t4,mod)
  # out2
  gen_f2add(t4,x0,x2,mod)
  gen_f2add(t5,y0,y2,mod)
  gen_f2mul(out2,t4,t5,mod)
  gen_f2sub(out2,out2,t0,mod)
  gen_f2sub(out2,out2,t2,mod)
  gen_f2add(out2,out2,t1,mod)

def gen_f6sqr(out,x,mod):
  gen_f6mul(out,x,x,mod)	# TODO: optimize


# f12

def gen_f12add(out,x,y,mod):
  global f12add_count
  f12add_count+=1
  print("// f6 add")
  x0 = x
  x1 = x0+288
  y0 = y
  y1 = y0+288
  out0 = out
  out1 = out0+288
  gen_f6add(out0,x0,y0,mod)
  gen_f6add(out1,x1,y1,mod)
  
def gen_f12sub(out,x,y,mod):
  global f12sub_count
  f12sub_count+=1
  print("// f6 add")
  x0 = x
  x1 = x0+288
  y0 = y
  y1 = y0+288
  out0 = out
  out1 = out0+288
  gen_f6sub(out0,x0,y0,mod)
  gen_f6sub(out1,x1,y1,mod)

def gen_f12mul(out,x,y,mod):
  global f12mul_count
  f12mul_count+=1
  print("// f12 mul")
  x0 = x
  x1 = x0+288
  y0 = y
  y1 = y0+288
  out0 = out
  out00 = out+0
  out01 = out+96
  out02 = out+192
  out1 = out0+288
  # temporary variables
  t0 = buffer_f12mul
  t00 = t0
  t01 = t0+96
  t02 = t0+192
  t1 = t0+288
  t10 = t1
  t11 = t1+96
  t12 = t1+192
  t2 = t1+576
  gen_f6mul(t0,x0,y0,mod)
  gen_f6mul(t1,x1,y1,mod)
  # out1
  gen_f6add(t2,x0,x1,mod)
  gen_f6add(out1,x0,x1,mod)
  gen_f6mul(out1,out1,t2,mod)
  gen_f6sub(out1,out1,t0,mod)
  gen_f6sub(out1,out1,t1,mod)
  # out0
  gen_mul_by_u_plus_1_fp2(t2,t12,mod)	# clobbers, so use temp
  gen_memcopy(t12,t2,96)		# since above clobbers
  gen_f2add(out00,t00,t12,mod)
  gen_f2add(out01,t01,t10,mod)
  gen_f2add(out02,t02,t11,mod)

def gen_f12sqr(out,x,mod):
  gen_f12mul(out,x,x,mod)		# TODO: optimize

def gen_f12_conjugate(x,mod):
  x1 = x+96
  gen_f6neg(x1,x1,mod)


# f6 and f12 optimizations for custom operations

def gen_mul_by_0y0_fp6(out,x,y,mod):
  # x is f6, y is f2
  x0 = x
  x1 = x0+96
  x2 = x1+96
  y0 = y
  y1 = y0+48
  out0 = out
  out1 = out0+96
  out2 = out1+96
  t = buffer_f6mul
  gen_f2mul(t,x2,y,mod)
  gen_f2mul(out2,x1,y,mod)
  gen_f2mul(out1,x0,y,mod)
  gen_mul_by_u_plus_1_fp2(out0,t,mod)
  
def gen_mul_by_xy0_fp6(out,x,y,mod):
  # x is f6, y is f6
  x0 = x
  x1 = x0+96
  x2 = x1+96
  y0 = y
  y1 = y0+96
  y2 = y1+96
  out0 = out
  out1 = out0+96
  out2 = out1+96
  t0 = buffer_f6mul
  t1 = t0+96
  t2 = t1+96	# unused
  t3 = t2+96
  t4 = t3+96
  t5 = t4+96
  gen_f2mul(t0,x0,y0,mod)
  gen_f2mul(t1,x1,y1,mod)
  gen_f2mul(t3,x2,y1,mod)
  gen_mul_by_u_plus_1_fp2(t3,t3,mod)	# TODO: maybe clobbers?
  gen_f2add(out0,x0,x1,mod)
  
  gen_f2add(t4,x0,x1,mod)
  gen_f2add(t5,y0,y1,mod)
  gen_f2mul(out1,t4,t5,mod)
  gen_f2sub(out1,out1,t0,mod)
  gen_f2sub(out1,out1,t1,mod)
  
  gen_f2mul(out2,x2,y0,mod)
  gen_f2add(out2,out2,t1,mod)

def gen_mul_by_xy00z0_fp12(out,x,y,mod):
  # x is f12, y is f6
  x0 = x
  x1 = x0+288
  y0 = y
  y1 = y0+96
  y2 = y1+96
  out0 = out
  out00 = out0
  out01 = out0+288
  out02 = out0+576
  out1 = out+864
  t0 = buffer_f12mul
  t00 = t0
  t01 = t00+96
  t02 = t01+96
  t1 = t0+288
  t10 = t1
  t11 = t10+96
  t12 = t11+96
  t2 = t1+576
  t20 = t2
  t21 = t2+96
  gen_mul_by_xy0_fp6(t0,x0,y,mod)
  gen_mul_by_0y0_fp6(t1,x1,y2,mod)
  gen_memcopy(t2,y,96)
  gen_f2add(t21,y1,y2,mod)
  gen_f6add(out1,x0,x1,mod)
  gen_mul_by_xy0_fp6(out1,out1,t2,mod)
  gen_f6sub(out1,out1,t0,mod)
  gen_f6sub(out1,out1,t1,mod)
  gen_mul_by_u_plus_1_fp2(t12,t12,mod)
  gen_f2add(out00,t00,t12,mod)
  gen_f2add(out01,t01,t10,mod)
  gen_f2add(out02,t02,t11,mod)
  




###############################
# Curve operations: add, double

def gen_Eadd__madd_2001_b(f,XYZout,XYZ1,XYZ2,mod):
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
  ZZ1 = buffer_Eadd
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

def gen_Eadd__madd_2007_bl(f,XYZout,XYZ1,XYZ2,mod):
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
  Z1Z1 = buffer_Eadd
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
  gen_fmul(f,J,Y1,J,mod)	# overwriting J		TODO: overwrite something else, since need J
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

  return I,J,r




def gen_Edouble__dbl_2009_alnr(f,XYZout,XYZ,mod):
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
  A = buffer_Edouble
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
  return A,B,E,F,ZZ,X1

def gen_Edouble__dbl_2009_l(f,XYZout,XYZ,mod):
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
  A = buffer_Edouble
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




#########
# Pairing

def gen_consts():
  # f12 one in mont form
  one = "15f65ec3fa80e4935c071a97a256ec6d77ce5853705257455f48985753c758baebf4000bc40c0002760900000002fffd"
  gen_memstore(f12one,bytes.fromhex(one))
  # prime
  p = "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab"
  gen_memstore(mod,bytes.fromhex(p))
  # inv
  inv="89f3fffcfffcfffd000000000000000000000000000000000000000000000000"
  gen_memstore(mod+48,bytes.fromhex(inv))

def gen_line_add(line,T,R,Q,mod):
  # T on E2, R on E2, Q on E2 affine
  TZ = T+384
  QX = Q
  QY = Q+192
  # ecadd
  I,J,r = gen_Eadd__madd_2007_bl("f2",T,R,Q,mod)
  # line eval
  gen_f2mul(I,r,QX,mod)
  gen_f2mul(J,QY,TZ,mod)
  gen_f2sub(I,I,J,mod)
  gen_f2add(line,I,I,mod)
  gen_memcopy(line+48,r,48)	# ??? I think this is correct
  gen_memcopy(line+96,TZ,96)	# ??? I think this is correct
  
def gen_line_dbl(line,T,Q,mod):
  # line is f6, T is on E2, Q is on E2
  line0 = line
  line1 = line0+96
  line2 = line1+96
  TZ = T+192
  # double
  A,B,E,F,ZZ,X1 = gen_Edouble__dbl_2009_alnr("f2",T,Q,mod)
  # eval line
  gen_f2add(line0,E,Q,mod)
  gen_f2sqr(line0,line0,mod)
  gen_f2sub(line0,line0,A,mod)
  gen_f2sub(line0,line0,F,mod)
  gen_f2add(B,B,B,mod)
  gen_f2add(B,B,B,mod)
  gen_f2sub(line0,line0,B,mod)
  gen_f2mul(line1,E,ZZ,mod)
  gen_f2mul(line2,TZ,ZZ,mod)
  
def gen_line_by_Px2(line,Px2,mod):
  # line is f6, Px2 is E2 point
  Px2X = Px2
  Px2Y = Px2X+96
  line00 = line
  line01 = line00+48
  line10 = line01+48
  line11 = line10+48
  line20 = line11+48
  line21 = line20+48
  gen_f1mul(line10,line10,Px2X,mod)
  gen_f1mul(line11,line11,Px2X,mod)
  gen_f1mul(line20,line20,Px2Y,mod)
  gen_f1mul(line21,line21,Px2Y,mod)

def gen_start_dbl(out,T,Px2,mod):
  out00 = out
  out11 = out+288+96	# ??
  line = buffer_line	# 3 f2 points
  line0 = line
  line2 = line0+192
  gen_line_dbl(line,T,T,mod)
  gen_line_by_Px2(line,Px2,mod)
  gen_memcopy(out,zero,576)
  gen_memcopy(out00,line0,192)
  gen_memcopy(out11,line2,96)
  # TODO: confirm that loop is never entered

def gen_add_dbl(out,T,Q,Px2,k,mod):
  line = buffer_line	# 3 f2 points
  gen_line_add(line,T,T,Q,mod)
  gen_line_by_Px2(line,Px2,mod)
  gen_mul_by_xy00z0_fp12(out,out,line,mod)
  # loop init	#TODO
  # put k on stack
  # while(k--)
  for i in range(k):
    gen_f12sqr(out,out,mod)
    gen_line_dbl(line,T,T,mod)
    gen_line_by_Px2(line,Px2,mod)
    gen_mul_by_xy00z0_fp12(out,out,line,mod)
  
  
  
def gen_miller_loop(out,P,Q,mod):
  # P is E1 point, Q is E2 point
  PX = P
  PY = P+48
  QX = Q
  # temp offsets
  T = buffer_miller_loop	# E2 point
  TX = T
  TY = TX+96
  TZ = TY+96
  Px2 = T+288			# E1 point, affine
  Px2X = Px2
  Px2Y = Px2+48
  # huff module
  print("#define macro MILLER_LOOP = takes(0) returns(0) {")
  # prepare some stuff
  gen_f1add(Px2,PX,PX,mod)
  gen_f1neg(Px2X,Px2X,mod)
  gen_f1add(Px2Y,PY,PY,mod)
  gen_memcopy(TX,QX,192)
  gen_memcopy(TZ,f12one,96)
  # execute
  gen_start_dbl(out,T,Px2,mod)
  gen_add_dbl(out,T,Q,Px2,2,mod)
  gen_add_dbl(out,T,Q,Px2,3,mod)
  gen_add_dbl(out,T,Q,Px2,9,mod)
  gen_add_dbl(out,T,Q,Px2,32,mod)
  gen_add_dbl(out,T,Q,Px2,16,mod)
  gen_f12_conjugate(out,mod)
  print("} // MILLER_LOOP")
  
def gen_final_exponentiation(out,in_):
  pass

def gen_pairing():
  #gen_consts()
  # input
  gen_miller_loop(buffer_output,buffer_inputs,buffer_inputs+144,mod)
  #gen_final_exponentiation(buffer_output,buffer_output)

  


  
if __name__=="__main__":
  #gen_Eadd__madd_2007_bl("f2",1,2,3,4)
  #gen_Edouble__dbl_2009_alnr("f2",1,2,3)
  #gen_Eadd__madd_2007_bl("f1",1,2,3,4)
  #gen_Edouble__dbl_2009_alnr("f2",1,2,3)
  #gen_Eadd__madd_2001_b("f2",1,2,3,4)
  #gen_Edouble__dbl_2009_l("f2",1,2,3)
  #gen_Eadd__madd_2007_bl("f2",1,2,3,4)
  print()
  print()
  addmod384_count=0
  submod384_count=0
  mulmodmont384_count=0
  gen_pairing()
  if 0:
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



