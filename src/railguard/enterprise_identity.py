"""OIDC access-token introspection against a configured enterprise IdP.

No discovery from token claims, no redirects, no cached active result. IdP failure
fails closed. Enterprise claims must be configured on the IdP client mapper.
"""
import math
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx


class IdentityError(Exception):
    pass


@dataclass(frozen=True)
class Identity:
    subject: str
    tenant: str
    roles: tuple[str, ...]
    claims: dict = field(repr=False)


class OIDC:
    def __init__(self, endpoint, issuer, audience, client_id, client_secret, *, transport=None):
        parts=urlsplit(endpoint)
        if parts.scheme!='https' or not parts.hostname or parts.username or parts.password or parts.fragment:
            raise ValueError('OIDC introspection requires a configured HTTPS endpoint')
        if not all([issuer,audience,client_id,client_secret]):
            raise ValueError('OIDC issuer, audience and client credentials are required')
        self.endpoint,self.issuer,self.audience=endpoint,issuer,audience
        self.client_id,self.client_secret=client_id,client_secret
        self.transport=transport

    def resolve(self, token):
        if not isinstance(token,str) or not 1 <= len(token) <= 16384:
            raise IdentityError('Invalid access token')
        try:
            with httpx.Client(timeout=5,follow_redirects=False,transport=self.transport) as client:
                response=client.post(self.endpoint,auth=(self.client_id,self.client_secret),
                                     data={'token':token,'token_type_hint':'access_token'})
            if response.status_code!=200 or len(response.content)>65536:raise ValueError()
            claims=response.json()
            aud=claims.get('aud',[]);aud=[aud] if isinstance(aud,str) else aud
            sub,tenant=claims.get('sub'),claims.get('tenant_id')
            if (claims.get('active') is not True or claims.get('iss')!=self.issuer
                or self.audience not in aud or type(claims.get('exp')) not in (int,float)
                or not math.isfinite(claims['exp']) or claims['exp']<=time.time() or not isinstance(sub,str) or not 1<=len(sub)<=128
                or not isinstance(tenant,str) or not 1<=len(tenant)<=128):raise ValueError()
            resource=claims.get('resource_access',{}).get(self.audience,{})
            roles=resource.get('roles',[])
            if not isinstance(roles,list) or any(not isinstance(x,str) for x in roles):raise ValueError()
            return Identity(sub,tenant,tuple(roles),claims)
        except (httpx.HTTPError,ValueError,TypeError,AttributeError):
            raise IdentityError('Invalid access token or identity provider unavailable') from None
