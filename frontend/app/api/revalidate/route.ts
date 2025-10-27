import { NextRequest, NextResponse } from 'next/server'
import { revalidatePath } from 'next/cache'
import crypto from 'crypto'

interface RevalidateRequest {
  operation: string
  operation_id: string
  paths: string[]
  timestamp: number
  source: string
}

interface LegacyRevalidateRequest {
  paths?: string[]
  secret?: string
}

function verifyHMACSignature(payload: RevalidateRequest, signature: string, timestamp: number): boolean {
  const hmacSecret = process.env.REVALIDATION_HMAC_SECRET
  if (!hmacSecret) {
    console.warn('REVALIDATION_HMAC_SECRET not configured')
    return false
  }
  
  // Vérifier que la requête n'est pas trop ancienne (5 minutes max)
  const now = Math.floor(Date.now() / 1000)
  if (Math.abs(now - timestamp) > 300) {
    console.warn('Request timestamp too old or too far in future')
    return false
  }
  
  try {
    // Reconstruire la chaîne à signer
    const payloadJson = JSON.stringify(payload, Object.keys(payload).sort())
    const signString = `${timestamp}:${payloadJson}`
    
    // Calculer HMAC-SHA256
    const expectedSignature = crypto
      .createHmac('sha256', hmacSecret)
      .update(signString)
      .digest('hex')
    
    // Comparaison time-safe
    return crypto.timingSafeEqual(
      Buffer.from(signature, 'hex'),
      Buffer.from(expectedSignature, 'hex')
    )
  } catch (error) {
    console.error('HMAC verification error:', error)
    return false
  }
}

function createResponseSignature(responseData: any): string {
  const hmacSecret = process.env.REVALIDATION_HMAC_SECRET
  if (!hmacSecret) return ''
  
  try {
    const responseJson = JSON.stringify(responseData, Object.keys(responseData).sort())
    return crypto
      .createHmac('sha256', hmacSecret)
      .update(responseJson)
      .digest('hex')
  } catch (error) {
    console.error('Response signature creation error:', error)
    return ''
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const signature = request.headers.get('x-signature')
    const timestampHeader = request.headers.get('x-timestamp')
    
    // Support for both new HMAC and legacy secret authentication
    const isHMACRequest = signature && timestampHeader && body.operation
    const isLegacyRequest = body.secret || request.nextUrl.searchParams.get('secret')
    
    if (isHMACRequest) {
      // New HMAC authentication
      const timestamp = parseInt(timestampHeader!)
      
      if (!verifyHMACSignature(body as RevalidateRequest, signature!, timestamp)) {
        return NextResponse.json({ 
          revalidated: false, 
          error: 'Invalid HMAC signature',
          timestamp: new Date().toISOString()
        }, { status: 401 })
      }
      
      const revalidateRequest = body as RevalidateRequest
      console.log(`🔄 HMAC revalidation request: ${revalidateRequest.operation_id}`)
      
      // Process revalidation
      const revalidatedPaths: string[] = []
      const errors: string[] = []
      
      for (const path of revalidateRequest.paths) {
        try {
          revalidatePath(path)
          revalidatedPaths.push(path)
          console.log(`✅ Revalidated: ${path}`)
        } catch (error) {
          console.error(`❌ Failed to revalidate ${path}:`, error)
          errors.push(`${path}: ${error instanceof Error ? error.message : 'Unknown error'}`)
        }
      }
      
      const responseData = {
        revalidated: true,
        operation_id: revalidateRequest.operation_id,
        paths: revalidatedPaths,
        errors: errors.length > 0 ? errors : undefined,
        timestamp: new Date().toISOString(),
        source: 'nextjs-frontend'
      }
      
      // Create response with signature
      const responseSignature = createResponseSignature(responseData)
      const response = NextResponse.json(responseData)
      
      if (responseSignature) {
        response.headers.set('X-Response-Signature', responseSignature)
      }
      
      return response
      
    } else if (isLegacyRequest) {
      // Legacy secret authentication
      const legacyBody = body as LegacyRevalidateRequest
      const secret = legacyBody.secret || request.nextUrl.searchParams.get('secret')
      
      if (!secret || secret !== process.env.REVALIDATION_SECRET) {
        return NextResponse.json({ 
          revalidated: false, 
          error: 'Invalid secret' 
        }, { status: 401 })
      }
      
      console.log('🔄 Legacy revalidation request')
      
      // Get paths to revalidate
      const paths = legacyBody.paths || ['/predictions/latest']
      
      // Revalidate each path
      const revalidatedPaths: string[] = []
      const errors: string[] = []
      
      for (const path of paths) {
        try {
          revalidatePath(path)
          revalidatedPaths.push(path)
        } catch (error) {
          console.error(`Failed to revalidate ${path}:`, error)
          errors.push(`${path}: ${error instanceof Error ? error.message : 'Unknown error'}`)
        }
      }
      
      return NextResponse.json({
        revalidated: true,
        paths: revalidatedPaths,
        errors: errors.length > 0 ? errors : undefined,
        timestamp: new Date().toISOString()
      })
      
    } else {
      return NextResponse.json({ 
        revalidated: false, 
        error: 'Missing authentication (HMAC signature or secret required)' 
      }, { status: 401 })
    }
    
  } catch (error) {
    console.error('Revalidation API error:', error)
    return NextResponse.json({
      revalidated: false,
      error: 'Internal server error',
      timestamp: new Date().toISOString()
    }, { status: 500 })
  }
}

export async function GET(request: NextRequest) {
  // Support for GET requests with query parameters
  const secret = request.nextUrl.searchParams.get('secret')
  const pathsParam = request.nextUrl.searchParams.get('paths')
  
  if (!secret || secret !== process.env.REVALIDATION_SECRET) {
    return NextResponse.json({ 
      revalidated: false, 
      error: 'Invalid secret' 
    }, { status: 401 })
  }
  
  const paths = pathsParam ? pathsParam.split(',') : ['/predictions/latest']
  
  const revalidatedPaths: string[] = []
  const errors: string[] = []
  
  for (const path of paths) {
    try {
      revalidatePath(path.trim())
      revalidatedPaths.push(path.trim())
    } catch (error) {
      console.error(`Failed to revalidate ${path}:`, error)
      errors.push(`${path}: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }
  
  return NextResponse.json({
    revalidated: true,
    paths: revalidatedPaths,
    errors: errors.length > 0 ? errors : undefined,
    timestamp: new Date().toISOString()
  })
}